package consumer

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/brandradar/pkg/domain"
	"github.com/brandradar/pkg/messaging"
	"github.com/brandradar/services/processor/filter"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/nats-io/nats.go/jetstream"
	"github.com/redis/go-redis/v9"
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

type brandSettings struct {
	Keywords   []string
	Exclusions []string
	RiskWords  []string
}

type Consumer struct {
	pool  *pgxpool.Pool
	js    jetstream.JetStream
	redis *redis.Client
}

func New(pool *pgxpool.Pool, js jetstream.JetStream, rdb *redis.Client) *Consumer {
	return &Consumer{pool: pool, js: js, redis: rdb}
}

func (c *Consumer) Run(ctx context.Context) error {
	cons := messaging.NewConsumer(c.js, messaging.ConsumerConfig{
		Stream:        messaging.StreamMentions,
		Subjects:      []string{"mentions.>"},
		Durable:       "processor",
		FilterSubject: messaging.SubjectMentionsRaw,
	})

	return cons.Run(ctx, func(ctx context.Context, data []byte) error {
		return c.handle(ctx, data)
	})
}

func (c *Consumer) handle(ctx context.Context, data []byte) error {
	var raw RawMention
	if err := json.Unmarshal(data, &raw); err != nil {
		slog.Warn("unmarshal mention", "error", err)
		return nil
	}

	if raw.ExternalID != "" {
		dup, err := c.isDuplicate(ctx, raw.SourceID, raw.ExternalID)
		if err != nil {
			slog.Warn("dedup check", "error", err)
		}
		if dup {
			slog.Debug("duplicate mention skipped", "external_id", raw.ExternalID)
			return nil
		}
	}

	bs := c.loadBrandSettings(ctx, raw.BrandID)

	result := filter.Analyze(raw.Title, raw.Content, bs.Keywords, bs.Exclusions, bs.RiskWords)
	if !result.Keep {
		slog.Debug("mention filtered out", "url", raw.URL)
		return nil
	}

	mentionID, err := c.insertMention(ctx, raw, result)
	if err != nil {
		return fmt.Errorf("insert mention: %w", err)
	}

	filtered := struct {
		MentionID        uuid.UUID `json:"mention_id"`
		ProjectID        uuid.UUID `json:"project_id"`
		SourceID         uuid.UUID `json:"source_id"`
		BrandID          uuid.UUID `json:"brand_id"`
		ExternalID       string    `json:"external_id"`
		URL              string    `json:"url"`
		Title            string    `json:"title"`
		Content          string    `json:"content"`
		Author           string    `json:"author,omitempty"`
		PublishedAt      time.Time `json:"published_at"`
		MatchedKeywords  []string  `json:"matched_keywords"`
		MatchedRiskWords []string  `json:"matched_risk_words"`
	}{
		MentionID:        mentionID,
		ProjectID:        raw.ProjectID,
		SourceID:         raw.SourceID,
		BrandID:          raw.BrandID,
		ExternalID:       raw.ExternalID,
		URL:              raw.URL,
		Title:            raw.Title,
		Content:          raw.Content,
		Author:           raw.Author,
		PublishedAt:      raw.PublishedAt,
		MatchedKeywords:  result.MatchedKeywords,
		MatchedRiskWords: result.MatchedRiskWords,
	}

	pub := messaging.NewPublisher(c.js)
	if err := pub.Publish(ctx, messaging.SubjectMentionsFiltered, filtered); err != nil {
		return fmt.Errorf("publish filtered: %w", err)
	}

	slog.Info("mention processed", "mention_id", mentionID, "url", raw.URL,
		"keywords", len(result.MatchedKeywords), "risk_words", len(result.MatchedRiskWords))
	return nil
}

func (c *Consumer) isDuplicate(ctx context.Context, sourceID uuid.UUID, externalID string) (bool, error) {
	if c.redis != nil {
		key := fmt.Sprintf("dedup:%s:%s", sourceID, externalID)
		exists, err := c.redis.Exists(ctx, key).Result()
		if err == nil && exists > 0 {
			return true, nil
		}
	}

	var count int
	err := c.pool.QueryRow(ctx,
		`SELECT COUNT(*) FROM mentions WHERE source_id = $1 AND external_id = $2`,
		sourceID, externalID,
	).Scan(&count)
	if err != nil {
		return false, err
	}

	if count > 0 {
		if c.redis != nil {
			key := fmt.Sprintf("dedup:%s:%s", sourceID, externalID)
			_ = c.redis.Set(ctx, key, "1", 24*time.Hour).Err()
		}
		return true, nil
	}

	return false, nil
}

func (c *Consumer) insertMention(ctx context.Context, raw RawMention, result filter.Result) (uuid.UUID, error) {
	var id uuid.UUID
	err := c.pool.QueryRow(ctx,
		`INSERT INTO mentions (project_id, source_id, brand_id, external_id, title, text, url,
		                       published_at, fetched_at, matched_keywords, matched_risk_words, status)
		 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), $9, $10, $11)
		 ON CONFLICT (source_id, external_id) DO UPDATE SET id = mentions.id
		 RETURNING id`,
		raw.ProjectID, raw.SourceID, raw.BrandID, raw.ExternalID,
		nullString(raw.Title), raw.Content, nullString(raw.URL),
		raw.PublishedAt, result.MatchedKeywords, result.MatchedRiskWords,
		domain.MentionStatusPendingML,
	).Scan(&id)
	if err != nil {
		return uuid.Nil, err
	}

	if c.redis != nil {
		key := fmt.Sprintf("dedup:%s:%s", raw.SourceID, raw.ExternalID)
		_ = c.redis.Set(ctx, key, "1", 24*time.Hour).Err()
	}

	return id, nil
}

func (c *Consumer) loadBrandSettings(ctx context.Context, brandID uuid.UUID) brandSettings {
	if c.redis != nil {
		key := fmt.Sprintf("brand:%s", brandID)
		data, err := c.redis.Get(ctx, key).Bytes()
		if err == nil {
			var bs brandSettings
			if json.Unmarshal(data, &bs) == nil {
				return bs
			}
		}
	}

	var bs brandSettings
	_ = c.pool.QueryRow(ctx,
		`SELECT keywords, exclusions, risk_words FROM brands WHERE id = $1`, brandID,
	).Scan(&bs.Keywords, &bs.Exclusions, &bs.RiskWords)

	if len(bs.RiskWords) == 0 {
		bs.RiskWords = domain.DefaultRiskWords
	}

	if c.redis != nil {
		key := fmt.Sprintf("brand:%s", brandID)
		data, _ := json.Marshal(bs)
		_ = c.redis.Set(ctx, key, data, 5*time.Minute).Err()
	}

	return bs
}

func nullString(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}
