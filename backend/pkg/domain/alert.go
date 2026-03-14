package domain

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
)

type AlertStatus string

const (
	AlertStatusPending AlertStatus = "pending"
	AlertStatusSent    AlertStatus = "sent"
	AlertStatusFailed  AlertStatus = "failed"
)

type AlertConfig struct {
	ID                   uuid.UUID       `json:"id" db:"id"`
	ProjectID            uuid.UUID       `json:"project_id" db:"project_id"`
	SpikeThreshold       int             `json:"spike_threshold" db:"spike_threshold"`
	SpikeWindowMinutes   int             `json:"spike_window_minutes" db:"spike_window_minutes"`
	CooldownMinutes      int             `json:"cooldown_minutes" db:"cooldown_minutes"`
	NotificationChannels json.RawMessage `json:"notification_channels" db:"notification_channels"`
	Enabled              bool            `json:"enabled" db:"enabled"`
	CreatedAt            time.Time       `json:"created_at" db:"created_at"`
	UpdatedAt            time.Time       `json:"updated_at" db:"updated_at"`
}

type Alert struct {
	ID               uuid.UUID       `json:"id" db:"id"`
	ProjectID        uuid.UUID       `json:"project_id" db:"project_id"`
	Type             string          `json:"type" db:"type"`
	MentionCount     int             `json:"mention_count" db:"mention_count"`
	Threshold        int             `json:"threshold" db:"threshold"`
	SentimentSummary json.RawMessage `json:"sentiment_summary,omitempty" db:"sentiment_summary"`
	Status           AlertStatus     `json:"status" db:"status"`
	SentAt           *time.Time      `json:"sent_at,omitempty" db:"sent_at"`
	CreatedAt        time.Time       `json:"created_at" db:"created_at"`
}
