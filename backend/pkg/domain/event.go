package domain

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
)

type Event struct {
	ID        uuid.UUID       `json:"id" db:"id"`
	ProjectID *uuid.UUID      `json:"project_id,omitempty" db:"project_id"`
	SourceID  *uuid.UUID      `json:"source_id,omitempty" db:"source_id"`
	Type      string          `json:"type" db:"type"`
	Message   string          `json:"message" db:"message"`
	Metadata  json.RawMessage `json:"metadata,omitempty" db:"metadata"`
	CreatedAt time.Time       `json:"created_at" db:"created_at"`
}
