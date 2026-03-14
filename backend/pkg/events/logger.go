package events

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Logger struct {
	pool *pgxpool.Pool
}

func NewLogger(pool *pgxpool.Pool) *Logger {
	return &Logger{pool: pool}
}

func (l *Logger) Log(ctx context.Context, projectID *uuid.UUID, sourceID *uuid.UUID, eventType, message string, metadata map[string]interface{}) error {
	meta, err := json.Marshal(metadata)
	if err != nil {
		meta = []byte("{}")
	}

	_, err = l.pool.Exec(ctx,
		`INSERT INTO events (project_id, source_id, type, message, metadata) VALUES ($1, $2, $3, $4, $5)`,
		projectID, sourceID, eventType, message, meta,
	)
	if err != nil {
		return fmt.Errorf("events: log: %w", err)
	}
	return nil
}
