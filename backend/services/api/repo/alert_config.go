package repo

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/brandradar/pkg/domain"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

var ErrAlertConfigNotFound = errors.New("repo: alert config not found")

type AlertConfigRepo struct {
	pool *pgxpool.Pool
}

func NewAlertConfigRepo(pool *pgxpool.Pool) *AlertConfigRepo {
	return &AlertConfigRepo{pool: pool}
}

func (r *AlertConfigRepo) GetByProject(ctx context.Context, projectID uuid.UUID) (*domain.AlertConfig, error) {
	var ac domain.AlertConfig
	err := r.pool.QueryRow(ctx,
		`SELECT id, project_id, spike_threshold, spike_window_minutes, cooldown_minutes,
		        notification_channels, enabled, created_at, updated_at
		 FROM alert_configs WHERE project_id = $1`,
		projectID,
	).Scan(&ac.ID, &ac.ProjectID, &ac.SpikeThreshold, &ac.SpikeWindowMinutes,
		&ac.CooldownMinutes, &ac.NotificationChannels, &ac.Enabled, &ac.CreatedAt, &ac.UpdatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrAlertConfigNotFound
		}
		return nil, fmt.Errorf("repo: get alert config: %w", err)
	}
	return &ac, nil
}

func (r *AlertConfigRepo) Upsert(ctx context.Context, projectID uuid.UUID, threshold, windowMinutes, cooldownMinutes int, channels json.RawMessage, enabled bool) (*domain.AlertConfig, error) {
	var ac domain.AlertConfig
	err := r.pool.QueryRow(ctx,
		`INSERT INTO alert_configs (project_id, spike_threshold, spike_window_minutes, cooldown_minutes, notification_channels, enabled)
		 VALUES ($1, $2, $3, $4, $5, $6)
		 ON CONFLICT (project_id)
		 DO UPDATE SET spike_threshold = $2, spike_window_minutes = $3, cooldown_minutes = $4,
		               notification_channels = $5, enabled = $6, updated_at = NOW()
		 RETURNING id, project_id, spike_threshold, spike_window_minutes, cooldown_minutes,
		           notification_channels, enabled, created_at, updated_at`,
		projectID, threshold, windowMinutes, cooldownMinutes, channels, enabled,
	).Scan(&ac.ID, &ac.ProjectID, &ac.SpikeThreshold, &ac.SpikeWindowMinutes,
		&ac.CooldownMinutes, &ac.NotificationChannels, &ac.Enabled, &ac.CreatedAt, &ac.UpdatedAt)
	if err != nil {
		return nil, fmt.Errorf("repo: upsert alert config: %w", err)
	}
	return &ac, nil
}
