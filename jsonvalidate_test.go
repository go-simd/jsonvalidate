package jsonvalidate

import (
	"encoding/json"
	"math/rand"
	"strings"
	"testing"
)

// validCases and invalidCases together cover every acceptance/rejection class
// encoding/json.Valid distinguishes. Each is cross-checked against the standard
// library in TestAgainstStdlib so the table can never drift from the gate.
var validCases = []string{
	// literals and whitespace
	"null", "true", "false", "  null  ", "\t\n\r null \t",
	// numbers
	"0", "-0", "1", "-1", "123", "1.5", "-1.5", "0.0", "1e10", "1E10",
	"1.5e+3", "1.5E-3", "1e-0", "1e+0", "0e0", "-0.0e0", "12345678901234567890",
	"1.234567890123456789", "1e1000", "-9223372036854775808",
	// strings (incl. all simple escapes, \u, and raw invalid UTF-8 which the
	// stdlib accepts inside strings)
	`""`, `"a"`, `"hello world"`, `"\""`, `"\\"`, `"\/"`, `"\b\f\n\r\t"`,
	`"A"`, `"\uD800"`, `"￿"`, `"ꯍꯍ"`, "\"\x7f\"",
	"\"\xff\"", "\"\xc0\xc0\"", `"tab\tnewline\n"`, "\"key with spaces\"",
	`"a long enough string to exercise the 16 byte simd block path"`,
	// arrays
	"[]", "[ ]", "[1]", "[1,2,3]", "[ 1 , 2 , 3 ]", "[null,true,false]",
	`["a","b","c"]`, "[[],[]]", "[[[[]]]]", `[{"a":1},{"b":2}]`,
	// objects
	"{}", "{ }", `{"a":1}`, `{"a":1,"b":2}`, `{ "a" : 1 }`, `{"":""}`,
	`{"nested":{"deep":{"deeper":[1,2,3]}}}`, `{"unicode":"é"}`,
	// big-ish documents to exercise the SIMD whitespace/string scans
	`{"name":"jsonvalidate","arches":["amd64","arm64","riscv64","loong64","ppc64le","s390x"],"simd":true,"count":6}`,
	"[                                                                  1]",
	`"                                  spaces inside the string token  "`,
}

var invalidCases = []string{
	// empty / whitespace only
	"", " ", "\t\n\r ",
	// truncated/garbled literals
	"nul", "nulll", "tru", "fals", "True", "NULL", "n", "t", "f",
	"nullx", "truex", "nullnull", "true false",
	// literals that start right but differ mid-way (per-byte mismatch path)
	"trUe", "falSe", "nuLl", "fXlse",
	// numbers
	"00", "01", "-", "+1", "1.", ".1", "1e", "1e+", "1e-", "1.2.3", "0x1",
	"1ee1", "--1", "1.e3", "1.0e", "Infinity", "NaN", "0x", "1.2.3.4",
	"- 1", "0 0", "1 2", "1,",
	// strings
	`"`, `"abc`, "\"\x01\"", "\"\x1f\"", "\"\n\"", "\"\t\"", `"\x41"`,
	`"\a"`, `"\u004"`, `"\u00gg"`, `"\uXYZW"`, `"\u"`, `"\"`, `"unterminated`,
	`"\`, `"esc\`,
	// structural
	"[", "]", "{", "}", "[1", "[1,", "[1,]", "[,]", "[,1]", "[1 2]",
	"[1,2", "]extra", `{"a"}`, `{"a":}`, `{"a":1`, `{"a":1,}`, `{1:2}`,
	`{"a":1 "b":2}`, `{:}`, `{,}`, `{"a":1,,"b":2}`, "[null,]", "[true,]",
	// object truncated at each step (covers the EOF guards in object)
	`{"a"`, `{"a":`, `{"a":1`, `{"a":1,`, `{"a" 1}`,
	// object key is a malformed string (exercises the key str() failure path)
	`{"a\q":1}`, `{"unterminated key`,
	// trailing junk
	"1 2 3", "nulltrue", "{}{}", "[][]", `1abc`,
	// raw bytes outside strings
	"\xff", "\x7f", "\xef\xbb\xbfnull", "\xc2\xa0null",
}

func TestValidTable(t *testing.T) {
	for _, c := range validCases {
		if !Valid([]byte(c)) {
			t.Errorf("Valid(%q) = false, want true", c)
		}
		if !json.Valid([]byte(c)) {
			t.Errorf("BUG IN TABLE: json.Valid(%q) = false; case is not actually valid", c)
		}
	}
}

func TestInvalidTable(t *testing.T) {
	for _, c := range invalidCases {
		if Valid([]byte(c)) {
			t.Errorf("Valid(%q) = true, want false", c)
		}
		if json.Valid([]byte(c)) {
			t.Errorf("BUG IN TABLE: json.Valid(%q) = true; case is not actually invalid", c)
		}
	}
}

// TestAgainstStdlib asserts Valid == json.Valid on the whole table.
func TestAgainstStdlib(t *testing.T) {
	for _, c := range append(append([]string{}, validCases...), invalidCases...) {
		if got, want := Valid([]byte(c)), json.Valid([]byte(c)); got != want {
			t.Errorf("Valid(%q) = %v, json.Valid = %v", c, got, want)
		}
	}
}

// TestNesting checks the exact depth boundary encoding/json enforces (10000 OK,
// 10001 rejected) for arrays, objects, and a mix.
func TestNesting(t *testing.T) {
	for _, tc := range []struct {
		name        string
		open, close string
		tail        string
	}{
		{"array", "[", "]", ""},
		{"object", `{"a":`, "}", "1"},
	} {
		for _, d := range []int{1, 100, 9999, 10000, 10001, 20000} {
			s := strings.Repeat(tc.open, d) + tc.tail + strings.Repeat(tc.close, d)
			if got, want := Valid([]byte(s)), json.Valid([]byte(s)); got != want {
				t.Errorf("%s depth %d: Valid=%v json.Valid=%v", tc.name, d, got, want)
			}
		}
	}
}

// TestScanKernels exercises the SIMD scan kernels directly against their scalar
// reference over inputs long enough to span multiple blocks at every alignment,
// including the no-match (run to end) path.
func TestScanKernels(t *testing.T) {
	bodies := []string{
		"",
		"short",
		strings.Repeat("x", 64),                 // no stop byte: runs to end
		strings.Repeat("x", 17) + `"`,           // quote after a full block
		strings.Repeat("y", 40) + `\`,           // backslash
		strings.Repeat("z", 33) + "\x01",        // control byte
		strings.Repeat(" ", 64),                 // all whitespace
		strings.Repeat(" ", 19) + "x",           // non-ws after blocks
		strings.Repeat("\t\n\r ", 10) + "value", // mixed whitespace
		strings.Repeat(" ", 16) + "\xff",        // high byte ends a space run
	}
	for _, body := range bodies {
		data := []byte(body)
		for off := 0; off <= len(data); off++ {
			if got, want := scanString(data, off), scanStringScalar(data, off); got != want {
				t.Errorf("scanString(%q, %d) = %d, want %d", body, off, got, want)
			}
			if got, want := skipSpace(data, off), skipSpaceScalar(data, off); got != want {
				t.Errorf("skipSpace(%q, %d) = %d, want %d", body, off, got, want)
			}
		}
	}
}

// TestRandomAgainstStdlib drives many deterministic pseudo-random inputs —
// random byte soup, near-JSON token streams (so strings/whitespace runs span
// multiple SIMD blocks at every alignment), and lightly-mutated valid documents
// — through Valid and asserts each verdict matches encoding/json.Valid. It runs
// under qemu on every arch, so it (not just the native -fuzz job) proves each
// SIMD kernel's lane and operand order on all six targets.
func TestRandomAgainstStdlib(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	tokens := []string{
		"{", "}", "[", "]", ":", ",", " ", "\t", "\n", "\r", "true", "false",
		"null", "0", "-1", "1.5e3", `"`, `\`, `\"`, `\n`, `\u00`, `\uD800`,
		"abc", "\x00", "\x1f", "\xff", "\xc2\xa0",
		strings.Repeat("x", 20), strings.Repeat(" ", 20),
		`"` + strings.Repeat("s", 20) + `"`,
	}
	check := func(b []byte) {
		if got, want := Valid(b), json.Valid(b); got != want {
			t.Fatalf("Valid(%q) = %v, json.Valid = %v", b, got, want)
		}
	}
	// random byte soup
	for i := 0; i < 20000; i++ {
		b := make([]byte, rng.Intn(70))
		for j := range b {
			b[j] = byte(rng.Intn(256))
		}
		check(b)
	}
	// near-JSON token streams
	for i := 0; i < 20000; i++ {
		var sb strings.Builder
		for n := rng.Intn(12); n >= 0; n-- {
			sb.WriteString(tokens[rng.Intn(len(tokens))])
		}
		check([]byte(sb.String()))
	}
	// mutated valid documents
	for i := 0; i < 20000; i++ {
		base := validCases[rng.Intn(len(validCases))]
		b := []byte(base)
		for m := rng.Intn(3); m >= 0; m-- {
			if len(b) == 0 {
				break
			}
			b[rng.Intn(len(b))] = byte(rng.Intn(256))
		}
		check(b)
	}
}

// FuzzValid is the gate: Valid must equal encoding/json.Valid on arbitrary
// bytes. It runs natively on amd64/arm64 and under qemu on the other four SIMD
// arches, so every kernel's lane and operand order is proven correct.
func FuzzValid(f *testing.F) {
	seeds := append(append([]string{}, validCases...), invalidCases...)
	seeds = append(seeds,
		strings.Repeat("[", 40)+strings.Repeat("]", 40),
		`{"a":[1,2,{"b":"`+strings.Repeat("c", 50)+`"}]}`,
		strings.Repeat(" ", 40)+"true",
	)
	for _, s := range seeds {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		if got, want := Valid(data), json.Valid(data); got != want {
			t.Fatalf("Valid(%q) = %v, encoding/json.Valid = %v", data, got, want)
		}
	})
}
