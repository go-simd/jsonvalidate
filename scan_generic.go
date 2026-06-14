//go:build !amd64 && !arm64 && !riscv64 && !loong64 && !ppc64le && !s390x

package jsonvalidate

// scanString returns the smallest index i >= off with data[i] == '"', '\\', or
// data[i] < 0x20, or len(data) if none. On arches with no SIMD kernel this is
// the whole implementation; elsewhere it finishes the kernel's sub-stride tail.
func scanString(data []byte, off int) int { return scanStringScalar(data, off) }

// skipSpace returns the smallest index i >= off with data[i] not one of the
// four JSON whitespace bytes, or len(data) if all remaining bytes are space.
func skipSpace(data []byte, off int) int { return skipSpaceScalar(data, off) }
