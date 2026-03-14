package consumer

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/brandradar/pkg/domain"
	"github.com/brandradar/pkg/messaging"
	"github.com/brandradar/services/spike/detector"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/nats-io/nats.go/jetstream"
	"github.com/redis/go-redis/v9"
)

type ReadyMention struct {
	MentionID      uuid.UUID `json:"mention_id"`
	ProjectID      uuid.UUID `json:"project_id"`
	SentimentLabel string    `json:"sentiment_label"`
	IsDuplicate    bool      `json:"is_duplicate"`
}

type SpikeAlert struct {
	ProjectID  uuid.UUID `json:"project_id"`
	Current    float64   `json:"current"`
	Mean       float64   `json:"mean"`
	StdDev     float64   `json:"std_dev"`
	Threshold  float64   `json:"threshold"`
	DetectedAt time.Time `json:"detected_at"`
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
		Durable:       "spike-detector",
		FilterSubject: messaging.SubjectMentionsReady,
	})

	return cons.Run(ctx, func(ctx context.Context, data []byte) error {
		return c.handle(ctx, data)
	})
}

func (c *Consumer) handle(ctx context.Context, data []byte) error {
	var rm ReadyMention
	if err := json.Unmarshal(data, &rm); err != nil {
		slog.Warn("unmarshal mention", "error", err)
		return nil
	}

	if rm.IsDuplicate {
		return nil
	}

	cfg, err := c.loadAlertConfig(ctx, rm.ProjectID)
	if err != nil {
		slog.Debug("no alert config for project", "project_id", rm.ProjectID, "error", err)
		return nil
	}
	if !cfg.Enabled {
		return nil
	}

	counterKey := fmt.Sprintf("spike:count:%s", rm.ProjectID)
	count, err := c.redis.Incr(ctx, counterKey).Result()
	if err != nil {
		slog.Warn("redis incr", "error", err)
		return nil
	}
	if count == 1 {
		_ = c.redis.Expire(ctx, counterKey, time.Duration(cfg.SpikeWindowMinutes)*time.Minute).Err()
	}

	window := time.Duration(cfg.SpikeWindowMinutes) * time.Minute
	counts, err := c.getHourlyCounts(ctx, rm.ProjectID, window)
	if err != nil {
		slog.Warn("get hourly counts", "error", err)
		return nil
	}

	if len(counts) < 2 {
		return nil
	}

	historical := counts[:len(counts)-1]
	current := counts[len(counts)-1]

	threshold := float64(cfg.SpikeThreshold) / 10.0
	stats := detector.ComputeStats(historical)
	if !detector.IsSpike(current, stats, threshold) {
		return nil
	}

	cooldownKey := fmt.Sprintf("spike:cooldown:%s", rm.ProjectID)
	if c.redis.Exists(ctx, cooldownKey).Val() > 0 {
		slog.Debug("spike cooldown active", "project_id", rm.ProjectID)
		return nil
	}

	_ = c.redis.Set(ctx, cooldownKey, "1", time.Duration(cfg.CooldownMinutes)*time.Minute).Err()

	alert := SpikeAlert{
		ProjectID:  rm.ProjectID,
		Current:    current,
		Mean:       stats.Mean,
		StdDev:     stats.StdDev,
		Threshold:  threshold,
		DetectedAt: time.Now().UTC(),
	}

	pub := messaging.NewPublisher(c.js)
	if err := pub.Publish(ctx, messaging.SubjectAlertsTrigger, alert); err != nil {
		return fmt.Errorf("publish spike alert: %w", err)
	}

	slog.Info("spike detected", "project_id", rm.ProjectID, "current", current, "mean", stats.Mean, "stddev", stats.StdDev)
	return nil
}

func (c *Consumer) loadAlertConfig(ctx context.Context, projectID uuid.UUID) (*domain.AlertConfig, error) {
	cacheKey := fmt.Sprintf("alert_config:%s", projectID)
	if c.redis != nil {
		data, err := c.redis.Get(ctx, cacheKey).Bytes()
		if err == nil {
			var cfg domain.AlertConfig
			if json.Unmarshal(data, &cfg) == nil {
				return &cfg, nil
			}
		}
	}

	var cfg domain.AlertConfig
	err := c.pool.QueryRow(ctx,
		`SELECT id, project_id, spike_threshold, spike_window_minutes,
		        cooldown_minutes, notification_channels, enabled
		 FROM alert_configs WHERE project_id = $1`,
		projectID,
	).Scan(&cfg.ID, &cfg.ProjectID, &cfg.SpikeThreshold, &cfg.SpikeWindowMinutes,
		&cfg.CooldownMinutes, &cfg.NotificationChannels, &cfg.Enabled)
	if err != nil {
		return nil, err
	}

	if c.redis != nil {
		data, _ := json.Marshal(cfg)
		_ = c.redis.Set(ctx, cacheKey, data, 2*time.Minute).Err()
	}

	return &cfg, nil
}

func (c *Consumer) getHourlyCounts(ctx context.Context, projectID uuid.UUID, window time.Duration) ([]float64, error) {
	rows, err := c.pool.Query(ctx,
		`SELECT COUNT(*) as cnt
		 FROM mentions
		 WHERE project_id = $1 AND created_at >= NOW() - $2::interval
		 GROUP BY date_trunc('hour', created_at)
		 ORDER BY date_trunc('hour', created_at)`,
		projectID, window.String(),
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var counts []float64
	for rows.Next() {
		var cnt int64
		if err := rows.Scan(&cnt); err != nil {
			return nil, err
		}
		counts = append(counts, float64(cnt))
	}
	return counts, rows.Err()
}
