package domain

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
)

type SourceType string

const (
	SourceTypeRSS      SourceType = "rss"
	SourceTypeWeb      SourceType = "web"
	SourceTypeTelegram SourceType = "telegram"
)

type SourceStatus string

const (
	SourceStatusActive   SourceStatus = "active"
	SourceStatusError    SourceStatus = "error"
	SourceStatusDisabled SourceStatus = "disabled"
)

type Source struct {
	ID            uuid.UUID       `json:"id" db:"id"`
	ProjectID     uuid.UUID       `json:"project_id" db:"project_id"`
	Type          SourceType      `json:"type" db:"type"`
	Name          string          `json:"name" db:"name"`
	URL           string          `json:"url" db:"url"`
	Config        json.RawMessage `json:"config" db:"config"`
	Status        SourceStatus    `json:"status" db:"status"`
	LastFetchedAt *time.Time      `json:"last_fetched_at,omitempty" db:"last_fetched_at"`
	LastError     *string         `json:"last_error,omitempty" db:"last_error"`
	CreatedAt     time.Time       `json:"created_at" db:"created_at"`
	UpdatedAt     time.Time       `json:"updated_at" db:"updated_at"`
}
