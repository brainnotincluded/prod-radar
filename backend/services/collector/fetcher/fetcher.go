package fetcher

import (
	"context"
	"time"

	"github.com/google/uuid"
)

type RawMention struct {
	SourceID    uuid.UUID `json:"source_id"`
	BrandID     uuid.UUID `json:"brand_id"`
	ProjectID   uuid.UUID `json:"project_id"`
	ExternalID  string    `json:"external_id"`
	URL         string    `json:"url"`
	Title       string    `json:"title"`
	Content     string    `json:"content"`
	Author      string    `json:"author,omitempty"`
	PublishedAt time.Time `json:"published_at"`
}

type FetchOpts struct {
	LastFetchedAt *time.Time
}

type Fetcher interface {
	Fetch(ctx context.Context, opts FetchOpts) ([]RawMention, error)
	Type() string
}
