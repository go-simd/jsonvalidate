//go:build ppc64le

package jsonvalidate

import (
	"encoding/json"
	"math/rand"
	"strings"
	"testing"

	"golang.org/x/sys/cpu"
)

// TestDispatchPPC64LE drives validation down BOTH ppc64le paths — the VSX scan
// kernels (scanStrVSX/skipWsVSX) and the pure-Go scalar scan — by toggling
// hasVSX, restoring it with defer. The fallback (hasVSX=false) is always safe.
// The kernel branch emits ISA-3.0 (POWER9) instructions (LXVB16X, CNTTZD) that
// SIGILL on POWER8, so it is forced on only when the host is genuinely POWER9+.
// Under the QEMU power9 CI target IsPOWER9 is true, so both branches are covered
// there. Both paths must agree with encoding/json.Valid, so the boolean verdict
// is identical regardless of which scan path ran.
func TestDispatchPPC64LE(t *testing.T) {
	saved := hasVSX
	defer func() { hasVSX = saved }()

	rng := rand.New(rand.NewSource(7))
	tokens := []string{
		"{", "}", "[", "]", ":", ",", " ", "\t", "\n", "\r", "true", "false",
		"null", "0", "-1", "1.5e3", `"`, `\`, `\"`, `\n`, `\u00`, `\uD800`,
		"abc", "\x00", "\x1f", "\xff", "\xc2\xa0",
		strings.Repeat("x", 20), strings.Repeat(" ", 20),
		`"` + strings.Repeat("s", 20) + `"`,
	}

	// corpus: table cases + token streams long enough to span multiple 16-byte
	// blocks at every alignment, plus the scan-kernel stress bodies.
	corpus := func() [][]byte {
		var out [][]byte
		for _, c := range append(append([]string{}, validCases...), invalidCases...) {
			out = append(out, []byte(c))
		}
		for i := 0; i < 2000; i++ {
			var sb strings.Builder
			for n := rng.Intn(12); n >= 0; n-- {
				sb.WriteString(tokens[rng.Intn(len(tokens))])
			}
			out = append(out, []byte(sb.String()))
		}
		for _, body := range []string{
			strings.Repeat("x", 64),
			strings.Repeat("x", 17) + `"`,
			strings.Repeat(" ", 64) + "v",
			`"` + strings.Repeat("s", 40) + `"`,
		} {
			out = append(out, []byte(body))
		}
		return out
	}()

	check := func(label string) {
		for _, b := range corpus {
			if got, want := Valid(b), json.Valid(b); got != want {
				t.Fatalf("%s Valid(%q) = %v, json.Valid = %v", label, b, got, want)
			}
			// Cross-check the dispatchers directly against their scalar oracles
			// so both scan branches are exercised at every offset.
			for off := 0; off <= len(b); off++ {
				if got, want := scanString(b, off), scanStringScalar(b, off); got != want {
					t.Fatalf("%s scanString(%q,%d)=%d want %d", label, b, off, got, want)
				}
				if got, want := skipSpace(b, off), skipSpaceScalar(b, off); got != want {
					t.Fatalf("%s skipSpace(%q,%d)=%d want %d", label, b, off, got, want)
				}
			}
		}
	}

	// Scalar fallback: always safe.
	hasVSX = false
	check("fallback")

	// Kernel: ISA-3.0 (POWER9) instructions SIGILL on POWER8, so only force the
	// VSX branch on a genuine POWER9+ host (true under QEMU power9 CI).
	if !cpu.PPC64.IsPOWER9 {
		t.Log("pre-POWER9 host; VSX scan kernels not exercised")
		return
	}
	hasVSX = true
	check("kernel")
}
