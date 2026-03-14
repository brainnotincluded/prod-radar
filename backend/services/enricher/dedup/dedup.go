package dedup

import (
	"crypto/sha256"
	"fmt"
	"strings"
)

func ContentHash(url, title, content string) string {
	normalized := strings.ToLower(strings.TrimSpace(url) + "|" + strings.TrimSpace(title) + "|" + strings.TrimSpace(content))
	h := sha256.Sum256([]byte(normalized))
	return fmt.Sprintf("%x", h)
}
