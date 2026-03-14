package repo

import (
	"context"
	"encoding/json"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

type AlertRow struct {
	ID               uuid.UUID        `json:"id"`
	ProjectID        uuid.UUID        `json:"project_id"`
	Type             string           `json:"type"`
	MentionCount     int              `json:"mention_count"`
	Threshold        int              `json:"threshold"`
	SentimentSummary json.RawMessage  `json:"sentiment_summary"`
	Status           string           `json:"status"`
	SentAt           *time.Time       `json:"sent_at,omitempty"`
	CreatedAt        time.Time        `json:"created_at"`
}

type AlertRepo struct {
	pool *pgxpool.Pool
}

func NewAlertRepo(pool *pgxpool.Pool) *AlertRepo {
	return &AlertRepo{pool: pool}
}

func (r *AlertRepo) ListByProject(ctx context.Context, projectID uuid.UUID, limit, offset int) ([]AlertRow, error) {
	rows, err := r.pool.Query(ctx,
		`SELECT id, project_id, type, mention_count, threshold, sentiment_summary, status, sent_at, created_at
		 FROM alerts WHERE project_id = $1
		 ORDER BY created_at DESC LIMIT $2 OFFSET $3`,
		projectID, limit, offset,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var alerts []AlertRow
	for rows.Next() {
		var a AlertRow
		if err := rows.Scan(&a.ID, &a.ProjectID, &a.Type, &a.MentionCount, &a.Threshold, &a.SentimentSummary, &a.Status, &a.SentAt, &a.CreatedAt); err != nil {
			return nil, err
		}
		alerts = append(alerts, a)
	}
	return alerts, rows.Err()
}
