package fetcher

import (
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestFetchOpts_NilLastFetchedAt(t *testing.T) {
	opts := FetchOpts{}
	if opts.LastFetchedAt != nil {
		t.Error("expected nil LastFetchedAt")
	}
}

func TestFetchOpts_WithLastFetchedAt(t *testing.T) {
	now := time.Now()
	opts := FetchOpts{LastFetchedAt: &now}
	if opts.LastFetchedAt == nil {
		t.Error("expected non-nil LastFetchedAt")
	}
	if !opts.LastFetchedAt.Equal(now) {
		t.Errorf("expected %v, got %v", now, *opts.LastFetchedAt)
	}
}

func TestRawMention_ExternalID(t *testing.T) {
	rm := RawMention{
		SourceID:    uuid.New(),
		BrandID:     uuid.New(),
		ProjectID:   uuid.New(),
		ExternalID:  "guid-123",
		URL:         "http://example.com/article",
		Title:       "Test",
		Content:     "Content",
		PublishedAt: time.Now(),
	}
	if rm.ExternalID == "" {
		t.Error("expected non-empty ExternalID")
	}
}

func TestRawMention_EmptyExternalID(t *testing.T) {
	rm := RawMention{}
	if rm.ExternalID != "" {
		t.Error("expected empty ExternalID for zero value")
	}
}
