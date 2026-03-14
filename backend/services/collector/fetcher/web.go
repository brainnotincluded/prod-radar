package fetcher

import (
	"context"
	"crypto/sha256"
	"fmt"
	"strings"
	"time"

	"github.com/gocolly/colly/v2"
	"github.com/google/uuid"
)

type WebFetcher struct {
	sourceID  uuid.UUID
	brandID   uuid.UUID
	projectID uuid.UUID
	targetURL string
	selector  string
}

func NewWebFetcher(sourceID, brandID, projectID uuid.UUID, targetURL, selector string) *WebFetcher {
	if selector == "" {
		selector = "article, .post, .entry, .news-item"
	}
	return &WebFetcher{
		sourceID:  sourceID,
		brandID:   brandID,
		projectID: projectID,
		targetURL: targetURL,
		selector:  selector,
	}
}

func (f *WebFetcher) Type() string { return "web" }

func (f *WebFetcher) Fetch(_ context.Context, _ FetchOpts) ([]RawMention, error) {
	var mentions []RawMention
	var fetchErr error

	c := colly.NewCollector(
		colly.UserAgent("BrandRadar-Collector/1.0"),
	)

	c.SetRequestTimeout(30 * time.Second)

	c.OnHTML(f.selector, func(e *colly.HTMLElement) {
		title := strings.TrimSpace(e.ChildText("h1, h2, h3, .title"))
		content := strings.TrimSpace(e.Text)
		link := e.ChildAttr("a", "href")
		if link == "" {
			link = e.Attr("href")
		}
		if link != "" && !strings.HasPrefix(link, "http") {
			link = e.Request.AbsoluteURL(link)
		}
		if link == "" {
			link = f.targetURL
		}

		if content == "" {
			return
		}

		extID := fmt.Sprintf("%x", sha256.Sum256([]byte(link)))

		mentions = append(mentions, RawMention{
			SourceID:    f.sourceID,
			BrandID:     f.brandID,
			ProjectID:   f.projectID,
			ExternalID:  extID,
			URL:         link,
			Title:       title,
			Content:     content,
			PublishedAt: time.Now(),
		})
	})

	c.OnError(func(r *colly.Response, err error) {
		fetchErr = fmt.Errorf("web: scrape %s: %w", f.targetURL, err)
	})

	if err := c.Visit(f.targetURL); err != nil {
		return nil, fmt.Errorf("web: visit %s: %w", f.targetURL, err)
	}

	if fetchErr != nil {
		return nil, fetchErr
	}

	return mentions, nil
}
