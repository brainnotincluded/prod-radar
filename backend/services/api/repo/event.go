package repo

import (
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

type EventRow struct {
	ID        uuid.UUID  `json:"id"`
	ProjectID *uuid.UUID `json:"project_id,omitempty"`
	SourceID  *uuid.UUID `json:"source_id,omitempty"`
	Type      string     `json:"type"`
	Message   string     `json:"message"`
	CreatedAt time.Time  `json:"created_at"`
}

type EventRepo struct {
	pool *pgxpool.Pool
}

func NewEventRepo(pool *pgxpool.Pool) *EventRepo {
	return &EventRepo{pool: pool}
}

func (r *EventRepo) ListByProject(ctx context.Context, projectID uuid.UUID, limit, offset int) ([]EventRow, error) {
	rows, err := r.pool.Query(ctx,
		`SELECT id, project_id, source_id, type, message, created_at
		 FROM events WHERE project_id = $1
		 ORDER BY created_at DESC LIMIT $2 OFFSET $3`,
		projectID, limit, offset,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var events []EventRow
	for rows.Next() {
		var e EventRow
		if err := rows.Scan(&e.ID, &e.ProjectID, &e.SourceID, &e.Type, &e.Message, &e.CreatedAt); err != nil {
			return nil, err
		}
		events = append(events, e)
	}
	return events, rows.Err()
}
