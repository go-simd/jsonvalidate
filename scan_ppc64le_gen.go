//go:build ignore

// Command gen produces scan_ppc64le.s with go-asmgen: the two SIMD byte-class
// scans (16-byte VSX blocks, POWER8+ baseline) that accelerate JSON validation.
//
// Each byte c is classified by a pair of 16-entry nibble LUTs: c is "in the set"
// iff (loLUT[c&0x0F] & hiLUT[c>>4]) != 0. The two lookups use VPERM (a 16-entry
// nibble table lookup, as in go-simd/utf8) over c&0x0F and (c>>4)&0x0F; VAND of
// the halves gives a per-byte marker.
//
// The marker vector is loaded/stored so that lane k corresponds to memory byte
// k exactly as in go-simd/matchlen: the block is read with LXVD2X (first
// doubleword = bytes 0..7, little-endian within each doubleword), the marker's
// two doublewords are moved to GPRs with MFVSRD, and CNTTZD (count trailing
// zeros, >>3) finds the lowest-address set byte. The high doubleword (bytes
// 8..15) is rotated in with VSLDOI $8. The 16-entry LUTs are loaded with LXVB16X
// (natural element order) so VPERM indexes them correctly, as proven for utf8.
//
//   - scanStr: stop set = {'"','\\', c < 0x20}; marker (lo&hi) is nonzero at
//     in-set bytes, scanned directly with CNTTZD.
//   - skipWs: stop at the first non-whitespace byte; the whitespace LUTs make
//     the marker nonzero at whitespace, so the bytes are inverted (NOR) before
//     CNTTZD to scan for the first non-whitespace.
//
// If no byte matches, the kernel returns the block-aligned end and the scalar
// reference finishes the tail. The position-dependent qemu FuzzValid /
// TestScanKernels tests are the gate that the lane and operand order are correct.
//
// Run: GOWORK=off go run scan_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

var strLoLUT = []byte{3, 3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 11, 3, 3, 3}
var strHiLUT = []byte{1, 2, 4, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
var wsLoLUT = []byte{2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0}
var wsHiLUT = []byte{1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

func repByte(x byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = x
	}
	return b
}

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("data"), abi.Scalar("off", abi.Int64)},
		[]abi.Arg{abi.Scalar("ret", abi.Int64)},
	)
}

func main() {
	f := emit.NewFile("ppc64le")

	strLo := f.Data("strLo_p", strLoLUT)
	strHi := f.Data("strHi_p", strHiLUT)
	wsLo := f.Data("wsLo_p", wsLoLUT)
	wsHi := f.Data("wsHi_p", wsHiLUT)
	c04 := f.Data("c04_p", repByte(0x04, 16)) // per-byte shift count for >>4
	c0F := f.Data("c0F_p", repByte(0x0F, 16))

	// emit builds a 16-byte/block kernel. invert=true (skipWs) NORs the marker
	// bytes (stop at the first non-member); invert=false (scanStr) scans the
	// marker directly.
	//
	// Registers: R4=base, R5=len, R7=off(cursor), R6 scratch ptr.
	// Vectors: V24=loLUT, V25=hiLUT, V26=0x04, V27=0x0F; V0..V5 scratch.
	emit := func(name, loTbl, hiTbl string, invert bool) {
		b := ppc64.NewFunc(name, sig(), 0)
		b.LoadArg("data_base", "R4").LoadArg("data_len", "R5").LoadArg("off", "R7")
		b.Raw("MOVD $%s+0(SB), R6", loTbl).Raw("LXVB16X (R0)(R6), VS56") // V24
		b.Raw("MOVD $%s+0(SB), R6", hiTbl).Raw("LXVB16X (R0)(R6), VS57") // V25
		b.Raw("MOVD $%s+0(SB), R6", c04).Raw("LXVB16X (R0)(R6), VS58")   // V26
		b.Raw("MOVD $%s+0(SB), R6", c0F).Raw("LXVB16X (R0)(R6), VS59")   // V27
		b.Label("loop")
		b.Raw("ADD $16, R7, R8")
		b.Raw("CMP R5, R8")
		b.Raw("BLT done") // len < off+16 -> tail
		b.Raw("ADD R4, R7, R9")
		b.Raw("LXVD2X (R0)(R9), VS32") // V0 = block (matchlen byte order)
		// lo = perm(loLUT, c & 0x0F)
		b.Raw("VAND V0, V27, V1")     // V1 = c & 0x0F
		b.Raw("VPERM V24, V24, V1, V2")
		// hi = perm(hiLUT, c >> 4)
		b.Raw("VSRB V0, V26, V3")     // V3 = c >> 4 (per byte)
		b.Raw("VPERM V25, V25, V3, V4")
		// marker = lo & hi
		b.Raw("VAND V2, V4, V5") // V5 = marker
		// first doubleword (memory bytes 0..7)
		b.Raw("MFVSRD VS37, R10") // R10 = marker bytes 0..7
		if invert {
			b.Raw("NOR R10, R10, R10")
		}
		b.Raw("CMP R10, $0")
		b.Raw("BNE lo")
		// second doubleword (memory bytes 8..15)
		b.Raw("VSLDOI $8, V5, V5, V6")
		b.Raw("MFVSRD VS38, R10")
		if invert {
			b.Raw("NOR R10, R10, R10")
		}
		b.Raw("CMP R10, $0")
		b.Raw("BNE hi")
		b.Raw("ADD $16, R7, R7")
		b.Raw("BR loop")
		b.Label("lo")
		b.Raw("CNTTZD R10, R12")
		b.Raw("SRD $3, R12, R12")
		b.Raw("ADD R7, R12, R12")
		b.StoreRet("R12", "ret")
		b.Raw("RET")
		b.Label("hi")
		b.Raw("CNTTZD R10, R12")
		b.Raw("SRD $3, R12, R12")
		b.Raw("ADD $8, R12, R12")
		b.Raw("ADD R7, R12, R12")
		b.StoreRet("R12", "ret")
		b.Raw("RET")
		b.Label("done")
		b.StoreRet("R7", "ret")
		b.Ret()
		f.Add(b.Func())
	}

	emit("scanStrVSX", strLo, strHi, false)
	emit("skipWsVSX", wsLo, wsHi, true)

	if err := os.WriteFile("scan_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote scan_ppc64le.s")
}
