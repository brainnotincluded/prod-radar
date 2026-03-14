package dedup

import "testing"

func TestContentHash_Deterministic(t *testing.T) {
	h1 := ContentHash("https://example.com", "Title", "Content")
	h2 := ContentHash("https://example.com", "Title", "Content")
	if h1 != h2 {
		t.Errorf("same input should produce same hash: %s != %s", h1, h2)
	}
}

func TestContentHash_DifferentInputs(t *testing.T) {
	h1 := ContentHash("https://example.com", "Title A", "Content A")
	h2 := ContentHash("https://example.com", "Title B", "Content B")
	if h1 == h2 {
		t.Error("different inputs should produce different hashes")
	}
}

func TestContentHash_CaseInsensitive(t *testing.T) {
	h1 := ContentHash("https://example.com", "TITLE", "CONTENT")
	h2 := ContentHash("https://example.com", "title", "content")
	if h1 != h2 {
		t.Errorf("hash should be case-insensitive: %s != %s", h1, h2)
	}
}

func TestContentHash_TrimSpaces(t *testing.T) {
	h1 := ContentHash("https://example.com", "Title", "Content")
	h2 := ContentHash("  https://example.com  ", "  Title  ", "  Content  ")
	if h1 != h2 {
		t.Errorf("hash should trim spaces: %s != %s", h1, h2)
	}
}

func TestContentHash_Length(t *testing.T) {
	h := ContentHash("url", "title", "content")
	if len(h) != 64 {
		t.Errorf("sha256 hex should be 64 chars, got %d", len(h))
	}
}

func TestContentHash_EmptyInputs(t *testing.T) {
	h := ContentHash("", "", "")
	if h == "" {
		t.Error("hash should not be empty even for empty inputs")
	}
	if len(h) != 64 {
		t.Errorf("sha256 hex should be 64 chars, got %d", len(h))
	}
}
