package repo

import (
	"context"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

type MentionRow struct {
	ID               uuid.UUID  `json:"id"`
	ProjectID        uuid.UUID  `json:"project_id"`
	SourceID         uuid.UUID  `json:"source_id"`
	BrandID          uuid.UUID  `json:"brand_id"`
	ExternalID       string     `json:"external_id"`
	Title            *string    `json:"title"`
	Text             string     `json:"text"`
	URL              *string    `json:"url"`
	Author           *string    `json:"author,omitempty"`
	PublishedAt      time.Time  `json:"published_at"`
	MatchedKeywords  []string   `json:"matched_keywords"`
	MatchedRiskWords []string   `json:"matched_risk_words"`
	SentimentLabel   *string    `json:"sentiment_label"`
	SentimentScore   *float64   `json:"sentiment_score"`
	IsDuplicate      bool       `json:"is_duplicate"`
	Status           string     `json:"status"`
	CreatedAt        time.Time  `json:"created_at"`
}

type MentionFilter struct {
	ProjectID  uuid.UUID
	BrandID    *uuid.UUID
	SourceID   *uuid.UUID
	Sentiment  *string
	Status     *string
	Since      *time.Time
	Until      *time.Time
	Search     *string
	HasRisk    *bool
	Limit      int
	Offset     int
}

type TimelinePoint struct {
	Bucket    time.Time `json:"bucket"`
	Count     int64     `json:"count"`
	Sentiment string    `json:"sentiment"`
}

type MentionRepo struct {
	pool *pgxpool.Pool
}

func NewMentionRepo(pool *pgxpool.Pool) *MentionRepo {
	return &MentionRepo{pool: pool}
}

func (r *MentionRepo) List(ctx context.Context, f MentionFilter) ([]MentionRow, error) {
	query := `SELECT id, project_id, source_id, brand_id, external_id, title, text, url, author,
		            published_at, matched_keywords, matched_risk_words,
		            sentiment_label, sentiment_score, is_duplicate, status, created_at
		FROM mentions WHERE project_id = $1`
	args := []any{f.ProjectID}
	idx := 2

	if f.BrandID != nil {
		query += " AND brand_id = $" + itoa(idx)
		args = append(args, *f.BrandID)
		idx++
	}
	if f.SourceID != nil {
		query += " AND source_id = $" + itoa(idx)
		args = append(args, *f.SourceID)
		idx++
	}
	if f.Sentiment != nil {
		query += " AND sentiment_label = $" + itoa(idx)
		args = append(args, *f.Sentiment)
		idx++
	}
	if f.Status != nil {
		query += " AND status = $" + itoa(idx)
		args = append(args, *f.Status)
		idx++
	}
	if f.Since != nil {
		query += " AND published_at >= $" + itoa(idx)
		args = append(args, *f.Since)
		idx++
	}
	if f.Until != nil {
		query += " AND published_at <= $" + itoa(idx)
		args = append(args, *f.Until)
		idx++
	}
	if f.Search != nil {
		query += " AND (title ILIKE '%' || $" + itoa(idx) + " || '%' OR text ILIKE '%' || $" + itoa(idx) + " || '%')"
		args = append(args, *f.Search)
		idx++
	}
	if f.HasRisk != nil && *f.HasRisk {
		query += " AND array_length(matched_risk_words, 1) > 0"
	}

	query += " ORDER BY published_at DESC"

	if f.Limit > 0 {
		query += " LIMIT $" + itoa(idx)
		args = append(args, f.Limit)
		idx++
	}
	if f.Offset > 0 {
		query += " OFFSET $" + itoa(idx)
		args = append(args, f.Offset)
	}

	rows, err := r.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var mentions []MentionRow
	for rows.Next() {
		var m MentionRow
		if err := rows.Scan(&m.ID, &m.ProjectID, &m.SourceID, &m.BrandID, &m.ExternalID,
			&m.Title, &m.Text, &m.URL, &m.Author, &m.PublishedAt,
			&m.MatchedKeywords, &m.MatchedRiskWords,
			&m.SentimentLabel, &m.SentimentScore, &m.IsDuplicate, &m.Status, &m.CreatedAt); err != nil {
			return nil, err
		}
		mentions = append(mentions, m)
	}
	return mentions, rows.Err()
}

func (r *MentionRepo) GetByID(ctx context.Context, projectID, mentionID uuid.UUID) (*MentionRow, error) {
	var m MentionRow
	err := r.pool.QueryRow(ctx,
		`SELECT id, project_id, source_id, brand_id, external_id, title, text, url, author,
		        published_at, matched_keywords, matched_risk_words,
		        sentiment_label, sentiment_score, is_duplicate, status, created_at
		 FROM mentions WHERE id = $1 AND project_id = $2`,
		mentionID, projectID,
	).Scan(&m.ID, &m.ProjectID, &m.SourceID, &m.BrandID, &m.ExternalID,
		&m.Title, &m.Text, &m.URL, &m.Author, &m.PublishedAt,
		&m.MatchedKeywords, &m.MatchedRiskWords,
		&m.SentimentLabel, &m.SentimentScore, &m.IsDuplicate, &m.Status, &m.CreatedAt)
	if err != nil {
		return nil, err
	}
	return &m, nil
}

func (r *MentionRepo) Duplicates(ctx context.Context, projectID, mentionID uuid.UUID) ([]MentionRow, error) {
	var clusterID *uuid.UUID
	err := r.pool.QueryRow(ctx,
		`SELECT cluster_id FROM mentions WHERE id = $1 AND project_id = $2`,
		mentionID, projectID,
	).Scan(&clusterID)
	if err != nil {
		return nil, err
	}

	if clusterID == nil {
		return []MentionRow{}, nil
	}

	rows, err := r.pool.Query(ctx,
		`SELECT id, project_id, source_id, brand_id, external_id, title, text, url, author,
		        published_at, matched_keywords, matched_risk_words,
		        sentiment_label, sentiment_score, is_duplicate, status, created_at
		 FROM mentions
		 WHERE project_id = $1 AND cluster_id = $2 AND id != $3
		 ORDER BY published_at DESC
		 LIMIT 50`,
		projectID, clusterID, mentionID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var mentions []MentionRow
	for rows.Next() {
		var m MentionRow
		if err := rows.Scan(&m.ID, &m.ProjectID, &m.SourceID, &m.BrandID, &m.ExternalID,
			&m.Title, &m.Text, &m.URL, &m.Author, &m.PublishedAt,
			&m.MatchedKeywords, &m.MatchedRiskWords,
			&m.SentimentLabel, &m.SentimentScore, &m.IsDuplicate, &m.Status, &m.CreatedAt); err != nil {
			return nil, err
		}
		mentions = append(mentions, m)
	}
	return mentions, rows.Err()
}

func (r *MentionRepo) Dismiss(ctx context.Context, projectID, mentionID uuid.UUID) error {
	_, err := r.pool.Exec(ctx,
		`UPDATE mentions SET status = 'dismissed' WHERE id = $1 AND project_id = $2`,
		mentionID, projectID,
	)
	return err
}

func (r *MentionRepo) Count(ctx context.Context, projectID uuid.UUID) (int64, error) {
	var count int64
	err := r.pool.QueryRow(ctx, `SELECT COUNT(*) FROM mentions WHERE project_id = $1`, projectID).Scan(&count)
	return count, err
}

func (r *MentionRepo) Timeline(ctx context.Context, projectID uuid.UUID, since, until time.Time, interval string) ([]TimelinePoint, error) {
	if interval != "hour" && interval != "day" {
		interval = "day"
	}

	rows, err := r.pool.Query(ctx,
		`SELECT date_trunc($4, published_at) AS bucket,
		        COALESCE(sentiment_label, 'unknown') AS sentiment,
		        COUNT(*) AS cnt
		 FROM mentions
		 WHERE project_id = $1 AND published_at >= $2 AND published_at <= $3
		 GROUP BY bucket, sentiment
		 ORDER BY bucket`,
		projectID, since, until, interval,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var points []TimelinePoint
	for rows.Next() {
		var p TimelinePoint
		if err := rows.Scan(&p.Bucket, &p.Sentiment, &p.Count); err != nil {
			return nil, err
		}
		points = append(points, p)
	}
	return points, rows.Err()
}

func itoa(i int) string {
	return strconv.Itoa(i)
}
