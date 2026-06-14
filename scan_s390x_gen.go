//go:build ignore

// Command gen produces scan_s390x.s with go-asmgen: the two SIMD byte-class
// scans (16-byte vector-facility blocks, z13+ baseline) that accelerate JSON
// validation.
//
// Each byte c is classified by a pair of 16-entry nibble LUTs: c is "in the set"
// iff (loLUT[c&0x0F] & hiLUT[c>>4]) != 0. The two lookups use VPERM (a 16-entry
// nibble table lookup, as in go-simd/utf8) over c&0x0F and (c>>4)&0x0F; VN of
// the halves gives a per-byte marker that is nonzero exactly at "stop" bytes.
//
// BIG-ENDIAN: s390x is the only big-endian target, but VL puts the lowest memory
// address in element 0 (the high lane). VPERM and the byte-wise classification
// are lane-local, so the marker's element k corresponds to memory byte k.
// VFENEBS (Vector Find Element Not Equal, byte, with CC) scans elements from 0
// upward against a zero vector: it sets CC=1 and writes the first nonzero byte's
// index into result byte 7, or CC=3 if all elements are zero. The index is thus
// already the lowest-memory-address stop byte — no endian fix-up needed.
//
//   - scanStr: stop set = {'"','\\', c < 0x20}; the marker (lo&hi) is nonzero at
//     in-set bytes, so VFENEBS against zero finds the first stop directly.
//   - skipWs: stop at the first non-whitespace byte; the whitespace LUTs make
//     (lo&hi) nonzero at whitespace, so VCEQB against zero first turns
//     non-whitespace bytes into 0xFF, then VFENEBS finds the first such byte.
//
// If no byte matches (CC=3), the kernel returns the block-aligned end and the
// scalar reference finishes the tail. The position-dependent qemu FuzzValid /
// TestScanKernels tests are the gate that the lane and operand order (including
// the big-endian LUTs) are correct.
//
// Run: GOWORK=off go run scan_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

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
	f := emit.NewFile("s390x")

	strLo := f.Data("strLo_z", strLoLUT)
	strHi := f.Data("strHi_z", strHiLUT)
	wsLo := f.Data("wsLo_z", wsLoLUT)
	wsHi := f.Data("wsHi_z", wsHiLUT)

	// emit builds a 16-byte/block kernel. invert=true (skipWs) turns the marker
	// into "non-member" via VCEQB against zero; invert=false (scanStr) uses
	// (lo&hi) directly. VFENEBS then finds the first nonzero stop byte.
	//
	// Registers: R2=base, R3=len, R4=off(cursor), R5 scratch ptr.
	// Vectors: V24=loLUT, V25=hiLUT, V30=zero; V0..V6 scratch.
	emit := func(name, loTbl, hiTbl string, invert bool) {
		b := s390x.NewFunc(name, sig(), 0)
		b.LoadArg("data_base", "R2").LoadArg("data_len", "R3").LoadArg("off", "R4")
		b.Raw("MOVD $%s+0(SB), R5", loTbl).Raw("VL (R5), V24")
		b.Raw("MOVD $%s+0(SB), R5", hiTbl).Raw("VL (R5), V25")
		b.Raw("VZERO V30")
		b.Label("loop")
		b.Raw("ADD $16, R4, R6")
		b.Raw("CMPBGT R6, R3, done") // off+16 > len -> tail
		b.Raw("ADD R2, R4, R7")
		b.Raw("VL (R7), V0") // block
		// lo = perm(loLUT, c & 0x0F): clear high nibble with a 4-bit shift pair.
		b.Raw("VESRLB $4, V0, V1") // V1 = c >> 4 (high nibble)
		b.Raw("VESLB $4, V1, V2")  // V2 = (c>>4)<<4
		b.Raw("VSB V2, V0, V3")    // V3 = c - ((c>>4)<<4) = c & 0x0F (low nibble)
		b.Raw("VPERM V24, V24, V3, V4")
		// hi = perm(hiLUT, c >> 4)
		b.Raw("VPERM V25, V25, V1, V5")
		// marker = lo & hi
		b.Raw("VN V4, V5, V6")
		if invert {
			b.Raw("VCEQB V6, V30, V6") // 0xFF where marker==0 (non-member)
		}
		// first nonzero marker byte, scanning element 0 upward
		b.Raw("VFENEBS V6, V30, V2") // CC=1 found (index in V2 byte 7), CC=3 all-zero
		b.Raw("BVS next")            // CC=3 -> no match this block
		b.Raw("VLGVB $7, V2, R8")    // R8 = index of first stop byte
		b.Raw("ADD R4, R8, R8")
		b.StoreRet("R8", "ret")
		b.Raw("RET")
		b.Label("next")
		b.Raw("ADD $16, R4, R4")
		b.Raw("BR loop")
		b.Label("done")
		b.StoreRet("R4", "ret")
		b.Ret()
		f.Add(b.Func())
	}

	emit("scanStrVX", strLo, strHi, false)
	emit("skipWsVX", wsLo, wsHi, true)

	if err := os.WriteFile("scan_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote scan_s390x.s")
}
