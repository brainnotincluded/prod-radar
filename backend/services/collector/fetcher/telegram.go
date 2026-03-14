package fetcher

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
)

var telegramMsgRe = regexp.MustCompile(`<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>`)
var telegramDateRe = regexp.MustCompile(`datetime="([^"]+)"`)

type TelegramFetcher struct {
	sourceID  uuid.UUID
	brandID   uuid.UUID
	projectID uuid.UUID
	channel   string
	client    *http.Client
}

func NewTelegramFetcher(sourceID, brandID, projectID uuid.UUID, channel string) *TelegramFetcher {
	return &TelegramFetcher{
		sourceID:  sourceID,
		brandID:   brandID,
		projectID: projectID,
		channel:   channel,
		client:    &http.Client{Timeout: 30 * time.Second},
	}
}

func (f *TelegramFetcher) Type() string { return "telegram" }

func (f *TelegramFetcher) Fetch(ctx context.Context, opts FetchOpts) ([]RawMention, error) {
	url := fmt.Sprintf("https://t.me/s/%s", f.channel)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("telegram: create request: %w", err)
	}
	req.Header.Set("User-Agent", "BrandRadar-Collector/1.0")

	resp, err := f.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("telegram: fetch %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("telegram: unexpected status %d for %s", resp.StatusCode, url)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("telegram: read body: %w", err)
	}

	html := string(body)
	messages := telegramMsgRe.FindAllStringSubmatch(html, -1)
	dates := telegramDateRe.FindAllStringSubmatch(html, -1)

	var mentions []RawMention
	for i, msg := range messages {
		content := strings.TrimSpace(msg[1])
		content = regexp.MustCompile(`<[^>]+>`).ReplaceAllString(content, "")

		published := time.Now()
		if i < len(dates) {
			if t, err := time.Parse(time.RFC3339, dates[i][1]); err == nil {
				published = t
			}
		}

		if opts.LastFetchedAt != nil && published.Before(*opts.LastFetchedAt) {
			continue
		}

		postURL := fmt.Sprintf("https://t.me/%s/%d", f.channel, i+1)
		extID := fmt.Sprintf("%s_%d", f.channel, i+1)

		mentions = append(mentions, RawMention{
			SourceID:    f.sourceID,
			BrandID:     f.brandID,
			ProjectID:   f.projectID,
			ExternalID:  extID,
			URL:         postURL,
			Title:       "",
			Content:     content,
			Author:      f.channel,
			PublishedAt: published,
		})
	}

	return mentions, nil
}
