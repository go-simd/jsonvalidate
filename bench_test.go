package jsonvalidate

import (
	"encoding/json"
	"strings"
	"testing"
)

// benchInputs are representative documents: a string-heavy object, a
// number/array-heavy document, and a whitespace-heavy (pretty-printed) blob —
// the cases where the SIMD string and whitespace scans do the most work.
var benchInputs = map[string][]byte{
	"strings": []byte(`{"name":"jsonvalidate","desc":"` +
		strings.Repeat("a fairly long string value that spans many SIMD blocks ", 8) +
		`","tags":["alpha","beta","gamma","delta","epsilon"]}`),
	"numbers": []byte("[" + strings.Repeat("-123.456e+7,", 200) + "0]"),
	"whitespace": []byte("[\n" + strings.Repeat("    1,\n", 200) + "    2\n]"),
}

func BenchmarkValid(b *testing.B) {
	for name, in := range benchInputs {
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(len(in)))
			for i := 0; i < b.N; i++ {
				if !Valid(in) {
					b.Fatal("unexpected invalid")
				}
			}
		})
	}
}

func BenchmarkStdlibValid(b *testing.B) {
	for name, in := range benchInputs {
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(len(in)))
			for i := 0; i < b.N; i++ {
				if !json.Valid(in) {
					b.Fatal("unexpected invalid")
				}
			}
		})
	}
}
