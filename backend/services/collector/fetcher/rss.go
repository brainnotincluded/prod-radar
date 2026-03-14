package fetcher

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/mmcdole/gofeed"
)

type RSSFetcher struct {
	sourceID  uuid.UUID
	brandID   uuid.UUID
	projectID uuid.UUID
	feedURL   string
	parser    *gofeed.Parser
}

func NewRSSFetcher(sourceID, brandID, projectID uuid.UUID, feedURL string) *RSSFetcher {
	return &RSSFetcher{
		sourceID:  sourceID,
		brandID:   brandID,
		projectID: projectID,
		feedURL:   feedURL,
		parser:    gofeed.NewParser(),
	}
}

func (f *RSSFetcher) Type() string { return "rss" }

func (f *RSSFetcher) Fetch(ctx context.Context, opts FetchOpts) ([]RawMention, error) {
	feed, err := f.parser.ParseURLWithContext(f.feedURL, ctx)
	if err != nil {
		return nil, fmt.Errorf("rss: parse feed %s: %w", f.feedURL, err)
	}

	var mentions []RawMention
	for _, item := range feed.Items {
		published := time.Now()
		if item.PublishedParsed != nil {
			published = *item.PublishedParsed
		} else if item.UpdatedParsed != nil {
			published = *item.UpdatedParsed
		}

		if opts.LastFetchedAt != nil && published.Before(*opts.LastFetchedAt) {
			continue
		}

		externalID := item.GUID
		if externalID == "" {
			externalID = item.Link
		}

		author := ""
		if item.Author != nil {
			author = item.Author.Name
		}

		content := item.Description
		if item.Content != "" {
			content = item.Content
		}

		mentions = append(mentions, RawMention{
			SourceID:    f.sourceID,
			BrandID:     f.brandID,
			ProjectID:   f.projectID,
			ExternalID:  externalID,
			URL:         item.Link,
			Title:       item.Title,
			Content:     content,
			Author:      author,
			PublishedAt: published,
		})
	}

	return mentions, nil
}
