package fetcher

import (
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestRawMention_Fields(t *testing.T) {
	m := RawMention{
		SourceID:    uuid.New(),
		BrandID:     uuid.New(),
		ProjectID:   uuid.New(),
		URL:         "https://example.com/article",
		Title:       "Test Article",
		Content:     "Some content about brands",
		Author:      "author",
		PublishedAt: time.Now(),
	}

	if m.URL == "" {
		t.Error("URL should not be empty")
	}
	if m.SourceID == uuid.Nil {
		t.Error("SourceID should not be nil")
	}
}

func TestRSSFetcher_Type(t *testing.T) {
	f := NewRSSFetcher(uuid.New(), uuid.New(), uuid.New(), "https://example.com/feed")
	if f.Type() != "rss" {
		t.Errorf("expected 'rss', got %q", f.Type())
	}
}

func TestWebFetcher_Type(t *testing.T) {
	f := NewWebFetcher(uuid.New(), uuid.New(), uuid.New(), "https://example.com", "")
	if f.Type() != "web" {
		t.Errorf("expected 'web', got %q", f.Type())
	}
}

func TestWebFetcher_DefaultSelector(t *testing.T) {
	f := NewWebFetcher(uuid.New(), uuid.New(), uuid.New(), "https://example.com", "")
	if f.selector == "" {
		t.Error("default selector should not be empty")
	}
}

func TestWebFetcher_CustomSelector(t *testing.T) {
	f := NewWebFetcher(uuid.New(), uuid.New(), uuid.New(), "https://example.com", ".custom")
	if f.selector != ".custom" {
		t.Errorf("expected '.custom', got %q", f.selector)
	}
}

func TestTelegramFetcher_Type(t *testing.T) {
	f := NewTelegramFetcher(uuid.New(), uuid.New(), uuid.New(), "testchannel")
	if f.Type() != "telegram" {
		t.Errorf("expected 'telegram', got %q", f.Type())
	}
}

func TestTelegramFetcher_Channel(t *testing.T) {
	f := NewTelegramFetcher(uuid.New(), uuid.New(), uuid.New(), "mychannel")
	if f.channel != "mychannel" {
		t.Errorf("expected 'mychannel', got %q", f.channel)
	}
}
