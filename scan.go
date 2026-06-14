package jsonvalidate

// scanStringScalar and skipSpaceScalar are the portable reference scans. They
// define the exact contract every SIMD kernel must reproduce, and finish the
// sub-stride tail after a kernel stops on a block boundary. Keeping them in one
// place (and exercising them directly in tests) means the SIMD paths only ever
// have to match this behaviour.

// scanStringScalar returns the smallest index i >= off with data[i] == '"',
// '\\', or data[i] < 0x20, or len(data) if none.
func scanStringScalar(data []byte, off int) int {
	for i := off; i < len(data); i++ {
		c := data[i]
		if c == '"' || c == '\\' || c < 0x20 {
			return i
		}
	}
	return len(data)
}

// skipSpaceScalar returns the smallest index i >= off with data[i] not one of
// ' ', '\t', '\n', '\r', or len(data) if all remaining bytes are whitespace.
func skipSpaceScalar(data []byte, off int) int {
	for i := off; i < len(data); i++ {
		switch data[i] {
		case ' ', '\t', '\n', '\r':
		default:
			return i
		}
	}
	return len(data)
}
