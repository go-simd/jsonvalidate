//go:build ignore

// Command gen produces scan_arm64.s with go-asmgen: the two SIMD byte-class
// scans (16-byte NEON blocks) that accelerate JSON validation.
//
// Each byte c is classified by a pair of 16-entry nibble LUTs: c is "in the set"
// iff (loLUT[c&0x0F] & hiLUT[c>>4]) != 0, computed with two VTBL lookups (over
// c&0x0F and (c>>4)&0x0F — both < 16 so VTBL never zeroes a lane) and a VAND.
//
//   - scanStr: stop set = {'"','\\', c < 0x20}; the stop marker is (lo&hi)
//     itself (nonzero exactly at in-set bytes).
//   - skipWs: stop at the first non-whitespace byte; the whitespace LUTs make
//     (lo&hi) nonzero at whitespace, so VCMEQ against zero turns non-whitespace
//     bytes into the 0xFF stop marker.
//
// The first nonzero marker byte is found by reading the two 64-bit halves to
// GPRs and using RBIT+CLZ (count trailing zero bits, >>3 = byte index), exactly
// as in go-simd/matchlen. If no marker is set the kernel returns the
// block-aligned end and the scalar reference finishes the tail.
//
// Run: GOWORK=off go run scan_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func repByte(x byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = x
	}
	return b
}

var strLoLUT = []byte{3, 3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 11, 3, 3, 3}
var strHiLUT = []byte{1, 2, 4, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
var wsLoLUT = []byte{2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0}
var wsHiLUT = []byte{1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("data"), abi.Scalar("off", abi.Int64)},
		[]abi.Arg{abi.Scalar("ret", abi.Int64)},
	)
}

func main() {
	f := emit.NewFile("arm64")

	strLo := f.Data("strLo_a", strLoLUT)
	strHi := f.Data("strHi_a", strHiLUT)
	wsLo := f.Data("wsLo_a", wsLoLUT)
	wsHi := f.Data("wsHi_a", wsHiLUT)
	c0F := f.Data("c0F_a", repByte(0x0F, 16))

	// emit builds a 16-byte/block kernel. invert=true (skipWs) turns the marker
	// into "non-member" via VCMEQ against zero; invert=false (scanStr) uses
	// (lo&hi) directly as the marker.
	//
	// Registers: R0=base, R1=len, R2=off(cursor); V0..V5 scratch;
	// V6=loLUT, V7=hiLUT, V8=0x0F mask, V9=zero.
	emit := func(name, loTbl, hiTbl string, invert bool) {
		b := arm64.NewFunc(name, sig(), 0)
		b.LoadArg("data_base", "R0").LoadArg("data_len", "R1").LoadArg("off", "R2")
		b.Raw("MOVD $%s+0(SB), R3", loTbl).Raw("VLD1 (R3), [V6.B16]")
		b.Raw("MOVD $%s+0(SB), R3", hiTbl).Raw("VLD1 (R3), [V7.B16]")
		b.Raw("MOVD $%s+0(SB), R3", c0F).Raw("VLD1 (R3), [V8.B16]")
		b.Raw("VEOR V9.B16, V9.B16, V9.B16") // zero
		b.Label("loop")
		b.Raw("ADD $16, R2, R4")
		b.Raw("CMP R1, R4")
		b.Raw("BGT done") // off+16 > len
		b.Raw("ADD R0, R2, R5")
		b.Raw("VLD1 (R5), [V0.B16]") // block
		// lo = tbl(loLUT, c & 0x0F)
		b.Raw("VAND V8.B16, V0.B16, V1.B16")
		b.Raw("VTBL V1.B16, [V6.B16], V2.B16")
		// hi = tbl(hiLUT, (c>>4) & 0x0F)
		b.Raw("VUSHR $4, V0.B16, V1.B16")
		b.Raw("VAND V8.B16, V1.B16, V1.B16")
		b.Raw("VTBL V1.B16, [V7.B16], V0.B16")
		// marker = lo & hi (nonzero where in set)
		b.Raw("VAND V2.B16, V0.B16, V0.B16")
		if invert {
			// stop at non-member: 0xFF where (lo&hi)==0
			b.Raw("VCMEQ V9.B16, V0.B16, V0.B16")
		}
		// first nonzero marker byte across the two 64-bit halves
		b.Raw("VMOV V0.D[0], R6")
		b.Raw("CBNZ R6, lo_half")
		b.Raw("VMOV V0.D[1], R6")
		b.Raw("CBNZ R6, hi_half")
		b.Raw("ADD $16, R2, R2")
		b.Raw("JMP loop")
		b.Label("lo_half")
		b.Raw("RBIT R6, R7")
		b.Raw("CLZ R7, R7")    // trailing-zero bit count
		b.Raw("LSR $3, R7, R7") // -> byte index
		b.Raw("ADD R2, R7, R7")
		b.StoreRet("R7", "ret")
		b.Raw("RET")
		b.Label("hi_half")
		b.Raw("RBIT R6, R7")
		b.Raw("CLZ R7, R7")
		b.Raw("LSR $3, R7, R7")
		b.Raw("ADD $8, R7, R7")
		b.Raw("ADD R2, R7, R7")
		b.StoreRet("R7", "ret")
		b.Raw("RET")
		b.Label("done")
		b.StoreRet("R2", "ret")
		b.Ret()
		f.Add(b.Func())
	}

	emit("scanStrNEON", strLo, strHi, false)
	emit("skipWsNEON", wsLo, wsHi, true)

	if err := os.WriteFile("scan_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote scan_arm64.s")
}
