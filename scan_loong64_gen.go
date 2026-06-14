//go:build ignore

// Command gen produces scan_loong64.s with go-asmgen: the two SIMD byte-class
// scans (16-byte LSX blocks) that accelerate JSON validation.
//
// Each byte c is classified by a pair of 16-entry nibble LUTs: c is "in the set"
// iff (loLUT[c&0x0F] & hiLUT[c>>4]) != 0. The two lookups use VSHUFB (vshuf.b),
// whose low-16 index lanes select from the vk operand — here the LUT — indexed
// by c&0x0F and (c>>4)&0x0F (both < 16). ANDing the halves gives a per-byte
// marker; the first marker byte is found by reading the two 64-bit halves to
// GPRs and CTZV (count trailing zeros, >>3 = byte index), as in go-simd/matchlen.
//
//   - scanStr: stop set = {'"','\\', c < 0x20}; the marker (lo&hi) is nonzero at
//     in-set bytes, used directly.
//   - skipWs: stop at the first non-whitespace byte; the whitespace LUTs make
//     the marker nonzero at whitespace, so VSEQB against zero turns
//     non-whitespace bytes into the 0xFF stop marker.
//
// If no marker is set the kernel returns the block-aligned end and the scalar
// reference finishes the tail, so the verdict can never diverge from
// encoding/json.Valid.
//
// Run: GOWORK=off go run scan_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
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
	f := emit.NewFile("loong64")

	strLo := f.Data("strLo_l", strLoLUT)
	strHi := f.Data("strHi_l", strHiLUT)
	wsLo := f.Data("wsLo_l", wsLoLUT)
	wsHi := f.Data("wsHi_l", wsHiLUT)
	_ = repByte // VSRLB/VANDB take immediates; no broadcast constants needed.

	// emit builds a 16-byte/block kernel. invert=true (skipWs) turns the marker
	// into "non-member" via VSEQB against zero; invert=false (scanStr) uses
	// (lo&hi) directly.
	//
	// Registers: R4=base, R5=len, R6=off(cursor); V regs scratch.
	// V6=loLUT, V7=hiLUT, V8=zero.
	emit := func(name, loTbl, hiTbl string, invert bool) {
		b := loong64.NewFunc(name, sig(), 0)
		b.LoadArg("data_base", "R4").LoadArg("data_len", "R5").LoadArg("off", "R6")
		b.Raw("MOVV $%s+0(SB), R7", loTbl).Raw("VMOVQ (R7), V6")
		b.Raw("MOVV $%s+0(SB), R7", hiTbl).Raw("VMOVQ (R7), V7")
		b.Raw("VXORV V8, V8, V8") // zero
		b.Label("loop")
		b.Raw("ADDV $16, R6, R8")
		b.Raw("BLT R5, R8, done") // len < off+16 -> tail
		b.Raw("ADDV R4, R6, R9")
		b.Raw("VMOVQ (R9), V0") // block
		// lo = shuf(loLUT, c & 0x0F)
		b.Raw("VANDB $15, V0, V1")
		b.Raw("VSHUFB V6, V6, V1, V2") // vk=loLUT selected for index<16
		// hi = shuf(hiLUT, (c>>4) & 0x0F)
		b.Raw("VSRLB $4, V0, V1")
		b.Raw("VANDB $15, V1, V1")
		b.Raw("VSHUFB V7, V7, V1, V3")
		// marker = lo & hi
		b.Raw("VANDV V2, V3, V0")
		if invert {
			b.Raw("VSEQB V8, V0, V0") // 0xFF where (lo&hi)==0
		}
		// first nonzero marker byte across the two 64-bit halves
		b.Raw("VMOVQ V0.V[0], R10")
		b.Raw("BNE R10, R0, lo_half")
		b.Raw("VMOVQ V0.V[1], R10")
		b.Raw("BNE R10, R0, hi_half")
		b.Raw("ADDV $16, R6, R6")
		b.Raw("JMP loop")
		b.Label("lo_half")
		b.Raw("CTZV R10, R11")
		b.Raw("SRLV $3, R11, R11")
		b.Raw("ADDV R6, R11, R11")
		b.StoreRet("R11", "ret")
		b.Raw("RET")
		b.Label("hi_half")
		b.Raw("CTZV R10, R11")
		b.Raw("SRLV $3, R11, R11")
		b.Raw("ADDV $8, R11, R11")
		b.Raw("ADDV R6, R11, R11")
		b.StoreRet("R11", "ret")
		b.Raw("RET")
		b.Label("done")
		b.StoreRet("R6", "ret")
		b.Ret()
		f.Add(b.Func())
	}

	emit("scanStrLSX", strLo, strHi, false)
	emit("skipWsLSX", wsLo, wsHi, true)

	if err := os.WriteFile("scan_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote scan_loong64.s")
}
