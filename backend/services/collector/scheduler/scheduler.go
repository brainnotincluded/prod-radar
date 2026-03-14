package scheduler

import (
	"context"
	"encoding/json"
	"log/slog"
	"time"

	"github.com/brandradar/pkg/domain"
	"github.com/brandradar/pkg/events"
	"github.com/brandradar/pkg/messaging"
	"github.com/brandradar/services/collector/fetcher"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/nats-io/nats.go/jetstream"
)

type Scheduler struct {
	pool     *pgxpool.Pool
	js       jetstream.JetStream
	interval time.Duration
	evLog    *events.Logger
}

func New(pool *pgxpool.Pool, js jetstream.JetStream, interval time.Duration) *Scheduler {
	return &Scheduler{
		pool:     pool,
		js:       js,
		interval: interval,
		evLog:    events.NewLogger(pool),
	}
}

func (s *Scheduler) Run(ctx context.Context) {
	ticker := time.NewTicker(s.interval)
	defer ticker.Stop()

	slog.Info("collector scheduler started", "interval", s.interval)

	s.subscribeSourcesChanged(ctx)
	s.tick(ctx)

	for {
		select {
		case <-ctx.Done():
			slog.Info("collector scheduler stopped")
			return
		case <-ticker.C:
			s.tick(ctx)
		}
	}
}

func (s *Scheduler) subscribeSourcesChanged(ctx context.Context) {
	go func() {
		consumer := messaging.NewConsumer(s.js, messaging.ConsumerConfig{
			Stream:        messaging.StreamSources,
			Subjects:      []string{messaging.SubjectSourcesChanged},
			Durable:       "collector-sources",
			FilterSubject: messaging.SubjectSourcesChanged,
		})
		_ = consumer.Run(ctx, func(_ context.Context, _ []byte) error {
			slog.Info("sources changed event received, triggering immediate tick")
			s.tick(ctx)
			return nil
		})
	}()
}

func (s *Scheduler) tick(ctx context.Context) {
	sources, err := s.loadActiveSources(ctx)
	if err != nil {
		slog.Error("load sources", "error", err)
		return
	}

	slog.Info("tick: processing sources", "count", len(sources))

	for _, src := range sources {
		f := s.buildFetcher(src)
		if f == nil {
			continue
		}

		opts := fetcher.FetchOpts{
			LastFetchedAt: src.LastFetchedAt,
		}

		mentions, err := f.Fetch(ctx, opts)
		if err != nil {
			slog.Warn("fetch failed", "source_id", src.ID, "type", src.Type, "error", err)
			s.updateSourceError(ctx, src.ID, err.Error())
			pID := src.ProjectID
			sID := src.ID
			_ = s.evLog.Log(ctx, &pID, &sID, "source_fetch_error", err.Error(), map[string]any{
				"source_type": string(src.Type),
				"source_name": src.Name,
			})
			continue
		}

		slog.Info("fetched mentions", "source_id", src.ID, "count", len(mentions))

		pub := messaging.NewPublisher(s.js)
		published := 0
		for _, m := range mentions {
			if err := pub.Publish(ctx, messaging.SubjectMentionsRaw, m); err != nil {
				slog.Warn("publish mention", "error", err)
			} else {
				published++
			}
		}

		s.updateSourceSuccess(ctx, src.ID)
		pID := src.ProjectID
		sID := src.ID
		_ = s.evLog.Log(ctx, &pID, &sID, "source_fetched", "mentions collected", map[string]any{
			"source_type": string(src.Type),
			"source_name": src.Name,
			"fetched":     len(mentions),
			"published":   published,
		})
	}
}

func (s *Scheduler) loadActiveSources(ctx context.Context) ([]domain.Source, error) {
	rows, err := s.pool.Query(ctx,
		`SELECT s.id, s.project_id, s.type, s.name, s.url, s.config, s.status,
		        s.last_fetched_at, s.last_error, s.created_at, s.updated_at
		 FROM sources s
		 JOIN brands b ON b.project_id = s.project_id
		 WHERE s.status = 'active'`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var sources []domain.Source
	for rows.Next() {
		var src domain.Source
		if err := rows.Scan(&src.ID, &src.ProjectID, &src.Type, &src.Name, &src.URL, &src.Config,
			&src.Status, &src.LastFetchedAt, &src.LastError, &src.CreatedAt, &src.UpdatedAt); err != nil {
			return nil, err
		}
		sources = append(sources, src)
	}
	return sources, rows.Err()
}

func (s *Scheduler) buildFetcher(src domain.Source) fetcher.Fetcher {
	brandID := s.getBrandID(context.Background(), src.ProjectID)

	switch src.Type {
	case domain.SourceTypeRSS:
		return fetcher.NewRSSFetcher(src.ID, brandID, src.ProjectID, src.URL)
	case domain.SourceTypeWeb:
		selector := ""
		if src.Config != nil {
			var cfg struct {
				Selector string `json:"selector"`
			}
			_ = json.Unmarshal(src.Config, &cfg)
			selector = cfg.Selector
		}
		return fetcher.NewWebFetcher(src.ID, brandID, src.ProjectID, src.URL, selector)
	case domain.SourceTypeTelegram:
		channel := src.URL
		if src.Config != nil {
			var cfg struct {
				Channel string `json:"channel"`
			}
			_ = json.Unmarshal(src.Config, &cfg)
			if cfg.Channel != "" {
				channel = cfg.Channel
			}
		}
		return fetcher.NewTelegramFetcher(src.ID, brandID, src.ProjectID, channel)
	default:
		slog.Warn("unknown source type", "type", src.Type, "source_id", src.ID)
		return nil
	}
}

func (s *Scheduler) getBrandID(ctx context.Context, projectID uuid.UUID) uuid.UUID {
	var brandID uuid.UUID
	_ = s.pool.QueryRow(ctx, `SELECT id FROM brands WHERE project_id = $1 LIMIT 1`, projectID).Scan(&brandID)
	return brandID
}

func (s *Scheduler) updateSourceSuccess(ctx context.Context, sourceID uuid.UUID) {
	_, _ = s.pool.Exec(ctx,
		`UPDATE sources SET last_fetched_at = NOW(), last_error = NULL, updated_at = NOW() WHERE id = $1`,
		sourceID)
}

func (s *Scheduler) updateSourceError(ctx context.Context, sourceID uuid.UUID, errMsg string) {
	_, _ = s.pool.Exec(ctx,
		`UPDATE sources SET last_error = $2, updated_at = NOW() WHERE id = $1`,
		sourceID, errMsg)
}
