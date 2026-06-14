//go:build ignore

// Command gen produces scan_riscv64.s with go-asmgen: the two SIMD byte-class
// scans (16-byte RVV blocks, VLEN>=128) that accelerate JSON validation.
//
// Each byte c is classified by a pair of 16-entry nibble LUTs: c is "in the set"
// iff (loLUT[c&0x0F] & hiLUT[c>>4]) != 0. The two table lookups are RVV gathers
// (VRGATHERVV picks table[index] per lane) over c&0x0F and (c>>4)&0x0F; ANDing
// the halves gives a per-lane marker. A vector compare builds a mask and
// vfirst.m (VFIRSTM) returns the first set lane (-1 if none).
//
//   - scanStr: stop set = {'"','\\', c < 0x20}; stop where marker != 0
//     (VMSNEVI $0).
//   - skipWs: stop at the first non-whitespace byte; the whitespace LUTs make
//     the marker nonzero at whitespace, so stop where marker == 0 (VMSEQVI $0).
//
// If no lane matches, the kernel advances 16 bytes; on running out of full
// blocks it returns the block-aligned end and the scalar reference finishes the
// tail, so the verdict can never diverge from encoding/json.Valid.
//
// Run: GOWORK=off go run scan_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
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
	f := emit.NewFile("riscv64")

	strLo := f.Data("strLo_r", strLoLUT)
	strHi := f.Data("strHi_r", strHiLUT)
	wsLo := f.Data("wsLo_r", wsLoLUT)
	wsHi := f.Data("wsHi_r", wsHiLUT)
	_ = repByte // LUTs are 16 bytes; no broadcast constants needed on RVV.

	// emit builds a 16-byte/block kernel. stopNonzero=true (scanStr) stops where
	// the marker is nonzero; false (skipWs) stops where it is zero.
	//
	// Registers: X5=base, X6=len, X7=off(cursor); V regs scratch.
	// V8=loLUT, V9=hiLUT loaded once.
	emit := func(name, loTbl, hiTbl string, stopNonzero bool) {
		b := riscv64.NewFunc(name, sig(), 0)
		b.LoadArg("data_base", "X5").LoadArg("data_len", "X6").LoadArg("off", "X7")
		b.Raw("VSETVLI $16, E8, M1, TA, MA, X8") // VL = 16 bytes
		b.Raw("MOV $%s+0(SB), X9", loTbl).Raw("VLE8V (X9), V8")
		b.Raw("MOV $%s+0(SB), X9", hiTbl).Raw("VLE8V (X9), V9")
		b.Label("loop")
		b.Raw("ADD $16, X7, X10")
		b.Raw("BLT X6, X10, done") // len < off+16 -> tail
		b.Raw("ADD X5, X7, X11")
		b.Raw("VLE8V (X11), V1") // block
		// lo = gather(loLUT, c & 0x0F)
		b.Raw("VANDVI $15, V1, V2")
		b.Raw("VRGATHERVV V2, V8, V3")
		// hi = gather(hiLUT, (c>>4) & 0x0F)
		b.Raw("VSRLVI $4, V1, V2")
		b.Raw("VANDVI $15, V2, V2")
		b.Raw("VRGATHERVV V2, V9, V4")
		// marker = lo & hi
		b.Raw("VANDVV V3, V4, V5")
		if stopNonzero {
			b.Raw("VMSNEVI $0, V5, V0") // mask: marker != 0
		} else {
			b.Raw("VMSEQVI $0, V5, V0") // mask: marker == 0
		}
		b.Raw("VFIRSTM V0, X12") // first set lane, or -1
		b.Raw("BGE X12, X0, found")
		b.Raw("ADD $16, X7, X7")
		b.Raw("JMP loop")
		b.Label("found")
		b.Raw("ADD X7, X12, X12")
		b.StoreRet("X12", "ret")
		b.Raw("RET")
		b.Label("done")
		b.StoreRet("X7", "ret")
		b.Ret()
		f.Add(b.Func())
	}

	emit("scanStrRVV", strLo, strHi, true)
	emit("skipWsRVV", wsLo, wsHi, false)

	if err := os.WriteFile("scan_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote scan_riscv64.s")
}
