package consumer

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strconv"
	"time"

	"github.com/brandradar/pkg/domain"
	"github.com/brandradar/pkg/events"
	"github.com/brandradar/pkg/messaging"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/nats-io/nats.go/jetstream"
)

type SpikeAlert struct {
	ProjectID  uuid.UUID `json:"project_id"`
	Current    float64   `json:"current"`
	Mean       float64   `json:"mean"`
	StdDev     float64   `json:"std_dev"`
	Threshold  float64   `json:"threshold"`
	DetectedAt time.Time `json:"detected_at"`
}

type notificationChannel struct {
	Type string `json:"type"`
	URL  string `json:"url,omitempty"`
}

type Consumer struct {
	pool      *pgxpool.Pool
	js        jetstream.JetStream
	evLog     *events.Logger
	maxRetry  int
}

func New(pool *pgxpool.Pool, js jetstream.JetStream) *Consumer {
	return &Consumer{
		pool:     pool,
		js:       js,
		evLog:    events.NewLogger(pool),
		maxRetry: 3,
	}
}

func (c *Consumer) Run(ctx context.Context) error {
	cons := messaging.NewConsumer(c.js, messaging.ConsumerConfig{
		Stream:        messaging.StreamAlerts,
		Subjects:      []string{"alerts.>"},
		Durable:       "alerter",
		FilterSubject: messaging.SubjectAlertsTrigger,
	})

	return cons.Run(ctx, func(ctx context.Context, data []byte) error {
		return c.handle(ctx, data)
	})
}

func (c *Consumer) handle(ctx context.Context, data []byte) error {
	var spike SpikeAlert
	if err := json.Unmarshal(data, &spike); err != nil {
		slog.Warn("unmarshal alert", "error", err)
		return nil
	}

	alertID, err := c.saveAlert(ctx, spike)
	if err != nil {
		return fmt.Errorf("save alert: %w", err)
	}

	channels := c.loadNotificationChannels(ctx, spike.ProjectID)

	var notifyErr error
	for attempt := 1; attempt <= c.maxRetry; attempt++ {
		notifyErr = c.notify(ctx, spike, channels)
		if notifyErr == nil {
			break
		}
		slog.Warn("notify attempt failed", "attempt", attempt, "error", notifyErr)
		time.Sleep(time.Duration(attempt) * time.Second)
	}

	if notifyErr != nil {
		c.updateAlertStatus(ctx, alertID, domain.AlertStatusFailed)
		pID := spike.ProjectID
		_ = c.evLog.Log(ctx, &pID, nil, "alert_notification_failed", notifyErr.Error(), map[string]any{
			"alert_id": alertID.String(),
		})
		return nil
	}

	c.updateAlertStatus(ctx, alertID, domain.AlertStatusSent)
	pID := spike.ProjectID
	_ = c.evLog.Log(ctx, &pID, nil, "alert_sent", formatAlertMessage(spike), map[string]any{
		"alert_id":      alertID.String(),
		"mention_count": spike.Current,
	})

	slog.Info("alert processed", "alert_id", alertID, "project_id", spike.ProjectID)
	return nil
}

func (c *Consumer) saveAlert(ctx context.Context, spike SpikeAlert) (uuid.UUID, error) {
	alertID := uuid.New()
	summary, _ := json.Marshal(map[string]float64{
		"current":   spike.Current,
		"mean":      spike.Mean,
		"std_dev":   spike.StdDev,
		"threshold": spike.Threshold,
	})

	_, err := c.pool.Exec(ctx,
		`INSERT INTO alerts (id, project_id, type, mention_count, threshold, sentiment_summary, status, created_at)
		 VALUES ($1, $2, 'spike', $3, $4, $5, $6, $7)`,
		alertID, spike.ProjectID,
		int(spike.Current), int(spike.Threshold*10),
		summary, domain.AlertStatusPending,
		spike.DetectedAt,
	)
	return alertID, err
}

func (c *Consumer) loadNotificationChannels(ctx context.Context, projectID uuid.UUID) []notificationChannel {
	var raw json.RawMessage
	err := c.pool.QueryRow(ctx,
		`SELECT notification_channels FROM alert_configs WHERE project_id = $1 AND enabled = true`,
		projectID,
	).Scan(&raw)
	if err != nil {
		return []notificationChannel{{Type: "log"}}
	}

	var channels []notificationChannel
	if json.Unmarshal(raw, &channels) != nil || len(channels) == 0 {
		return []notificationChannel{{Type: "log"}}
	}
	return channels
}

func (c *Consumer) notify(_ context.Context, spike SpikeAlert, channels []notificationChannel) error {
	msg := formatAlertMessage(spike)
	for _, ch := range channels {
		switch ch.Type {
		case "webhook":
			slog.Info("webhook notification", "url", ch.URL, "message", msg)
		case "email":
			slog.Info("email notification", "to", ch.URL, "message", msg)
		case "telegram":
			slog.Info("telegram notification", "chat", ch.URL, "message", msg)
		default:
			slog.Info("log notification", "project_id", spike.ProjectID, "message", msg)
		}
	}
	return nil
}

func (c *Consumer) updateAlertStatus(ctx context.Context, alertID uuid.UUID, status domain.AlertStatus) {
	var sentAt *time.Time
	if status == domain.AlertStatusSent {
		now := time.Now()
		sentAt = &now
	}
	_, err := c.pool.Exec(ctx,
		`UPDATE alerts SET status = $2, sent_at = $3 WHERE id = $1`,
		alertID, status, sentAt,
	)
	if err != nil {
		slog.Warn("update alert status", "error", err)
	}
}

func formatAlertMessage(spike SpikeAlert) string {
	return "Spike detected: current=" +
		strconv.FormatFloat(spike.Current, 'f', 1, 64) +
		", mean=" + strconv.FormatFloat(spike.Mean, 'f', 1, 64) +
		", stddev=" + strconv.FormatFloat(spike.StdDev, 'f', 1, 64)
}
