//go:build ignore

// Command gen produces scan_amd64.s with go-asmgen: the two SIMD byte-class
// scans that accelerate JSON validation, each in an SSE2/SSSE3 (16 bytes/block)
// and an AVX2 (32 bytes/block) flavour.
//
// Both scans classify each byte by a pair of 16-entry nibble lookup tables: a
// byte c is "in the set" iff (loLUT[c&0x0F] & hiLUT[c>>4]) != 0. The lookups are
// PSHUFB over (c&0x0F) and ((c>>4)&0x0F) — both have bit 7 clear so PSHUFB never
// zeroes a lane spuriously. ANDing the two halves and comparing != 0 gives a
// per-byte 0xFF/0x00 mask; PMOVMSKB + TZCNT/BSF finds the first set byte.
//
//   - scanStr*: stop set = {'"', '\\', bytes < 0x20}. Returns the first such
//     index at/after off, or the block-aligned end of the scanned region.
//   - skipWs*: the LUTs detect JSON whitespace ({space,tab,nl,cr}); the kernel
//     stops at the first byte that is NOT whitespace (mask == 0), returning its
//     index, or the block-aligned end.
//
// Each kernel only ever returns an index the scalar reference (scanStringScalar
// / skipSpaceScalar) would also have returned, so the validator's verdict can
// never diverge from encoding/json.Valid; FuzzValid is the gate.
//
// Run: GOWORK=off go run scan_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func repByte(x byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = x
	}
	return b
}
func rep(v []byte, times int) []byte {
	var b []byte
	for i := 0; i < times; i++ {
		b = append(b, v...)
	}
	return b
}

// stop set = {'"'(0x22), '\\'(0x5C), c < 0x20}.
var strLoLUT = []byte{3, 3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 11, 3, 3, 3}
var strHiLUT = []byte{1, 2, 4, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

// whitespace set = {space(0x20), tab(0x09), nl(0x0A), cr(0x0D)}.
var wsLoLUT = []byte{2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0}
var wsHiLUT = []byte{1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("data"), abi.Scalar("off", abi.Int64)},
		[]abi.Arg{abi.Scalar("ret", abi.Int64)},
	)
}

func main() {
	f := emit.NewFile("amd64")

	// 16-byte constant tables (SSE) and 32-byte (AVX2, suffix B).
	strLo := f.Data("strLo", strLoLUT)
	strHi := f.Data("strHi", strHiLUT)
	wsLo := f.Data("wsLo", wsLoLUT)
	wsHi := f.Data("wsHi", wsHiLUT)
	c0F := f.Data("c0F", repByte(0x0F, 16))
	strLoB := f.Data("strLoB", rep(strLoLUT, 2))
	strHiB := f.Data("strHiB", rep(strHiLUT, 2))
	wsLoB := f.Data("wsLoB", rep(wsLoLUT, 2))
	wsHiB := f.Data("wsHiB", rep(wsHiLUT, 2))
	c0Fb := f.Data("c0Fb", repByte(0x0F, 32))

	// emitSSE builds a 16-byte/block kernel. stopOnMatch: true => stop where the
	// class mask is set (scanStr); false => stop where it is clear (skipWs).
	//
	// Registers: SI=base, DX=len, AX=cursor(off), scratch X0..X5, GP R8..R11.
	emitSSE := func(name, loTbl, hiTbl string, stopOnMatch bool) {
		b := amd64.NewFunc(name, sig(), 0)
		b.LoadArg("data_base", "SI").LoadArg("data_len", "DX").LoadArg("off", "AX")
		b.Raw("MOVOU %s+0(SB), X3", loTbl) // loLUT
		b.Raw("MOVOU %s+0(SB), X4", hiTbl) // hiLUT
		b.Raw("MOVOU %s+0(SB), X5", c0F)   // 0x0F mask
		b.Label("loop")
		// if off+16 > len: done (return current off, the block-aligned end)
		b.Raw("MOVQ AX, R8")
		b.Raw("ADDQ $16, R8")
		b.Raw("CMPQ R8, DX")
		b.Raw("JGT done")
		b.Raw("MOVOU (SI)(AX*1), X0") // block
		// lo = pshufb(loLUT, c & 0x0F)
		b.Raw("MOVO X0, X1")
		b.Raw("PAND X5, X1")
		b.Raw("MOVO X3, X2")
		b.Raw("PSHUFB X1, X2") // X2 = loLUT[c&0x0F]
		// hi = pshufb(hiLUT, (c>>4)&0x0F)
		b.Raw("MOVO X0, X1")
		b.Raw("PSRLW $4, X1")
		b.Raw("PAND X5, X1")
		b.Raw("MOVO X4, X0")
		b.Raw("PSHUFB X1, X0") // X0 = hiLUT[c>>4]
		// match = (lo & hi); 0xFF lanes where (lo&hi)!=0 after cmpeq trick.
		b.Raw("PAND X2, X0") // X0 = lo & hi (nonzero where in set)
		// build 0xFF mask where X0 != 0: compare-not-equal-to-zero via PCMPEQB 0
		// then invert. Simpler: PMOVMSKB of (X0 != 0). We get a per-byte 0x00
		// where lane==0; for nonzero lanes the byte is some nonzero value but
		// PMOVMSKB only reads the top bit. So normalise: cmpeq with zero gives
		// 0xFF where ==0, 0x00 where !=0; XOR all-ones to flip.
		b.Raw("PXOR X1, X1")
		b.Raw("PCMPEQB X1, X0") // 0xFF where lane==0 (not in set), 0x00 where in set
		b.Raw("PMOVMSKB X0, R9")
		if stopOnMatch {
			// in-set lanes => bit 0 in R9. We want first in-set => first 0 bit.
			// Invert low 16 bits then TZCNT.
			b.Raw("NOTL R9")
			b.Raw("ANDL $0xFFFF, R9")
		} else {
			// skipWs: stop at first non-whitespace. X0 currently has 0xFF where
			// lane is NOT whitespace (==0 in ws-class), so PMOVMSKB bit set =>
			// non-ws. First non-ws = first set bit.
			b.Raw("ANDL $0xFFFF, R9")
		}
		b.Raw("TESTL R9, R9")
		b.Raw("JZ next")
		b.Raw("BSFL R9, R9") // index of first set bit within block
		b.Raw("ADDQ R9, AX")
		b.StoreRet("AX", "ret")
		b.Ret()
		b.Label("next")
		b.Raw("ADDQ $16, AX")
		b.Raw("JMP loop")
		b.Label("done")
		b.StoreRet("AX", "ret")
		b.Ret()
		f.Add(b.Func())
	}

	// emitAVX2 is the 32-byte/block flavour.
	emitAVX2 := func(name, loTbl, hiTbl string, stopOnMatch bool) {
		b := amd64.NewFunc(name, sig(), 0)
		b.LoadArg("data_base", "SI").LoadArg("data_len", "DX").LoadArg("off", "AX")
		b.Raw("VMOVDQU %s+0(SB), Y3", loTbl)
		b.Raw("VMOVDQU %s+0(SB), Y4", hiTbl)
		b.Raw("VMOVDQU %s+0(SB), Y5", c0Fb)
		b.Raw("VPXOR Y6, Y6, Y6")
		b.Label("loop")
		b.Raw("MOVQ AX, R8")
		b.Raw("ADDQ $32, R8")
		b.Raw("CMPQ R8, DX")
		b.Raw("JGT done")
		b.Raw("VMOVDQU (SI)(AX*1), Y0")
		// VPSHUFB is per-128-bit-lane; both LUT halves are duplicated across the
		// two lanes (rep(...,2)) so the lookup is correct in each lane.
		b.Raw("VPAND Y5, Y0, Y1")     // c & 0x0F
		b.Raw("VPSHUFB Y1, Y3, Y2")   // loLUT[c&0x0F]
		b.Raw("VPSRLW $4, Y0, Y1")    // c >> 4
		b.Raw("VPAND Y5, Y1, Y1")     // (c>>4)&0x0F
		b.Raw("VPSHUFB Y1, Y4, Y0")   // hiLUT[c>>4]
		b.Raw("VPAND Y2, Y0, Y0")     // lo & hi
		b.Raw("VPCMPEQB Y6, Y0, Y0")  // 0xFF where ==0 (not in set)
		b.Raw("VPMOVMSKB Y0, R9")
		if stopOnMatch {
			b.Raw("NOTL R9")
		}
		b.Raw("TESTL R9, R9")
		b.Raw("JZ next")
		b.Raw("BSFL R9, R9")
		b.Raw("ADDQ R9, AX")
		b.Raw("VZEROUPPER")
		b.StoreRet("AX", "ret")
		b.Ret()
		b.Label("next")
		b.Raw("ADDQ $32, AX")
		b.Raw("JMP loop")
		b.Label("done")
		b.Raw("VZEROUPPER")
		b.StoreRet("AX", "ret")
		b.Ret()
		f.Add(b.Func())
	}

	emitSSE("scanStrSSE", strLo, strHi, true)
	emitSSE("skipWsSSE", wsLo, wsHi, false)
	emitAVX2("scanStrAVX2", strLoB, strHiB, true)
	emitAVX2("skipWsAVX2", wsLoB, wsHiB, false)

	if err := os.WriteFile("scan_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote scan_amd64.s")
}
