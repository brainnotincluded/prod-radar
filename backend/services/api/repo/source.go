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

var ErrSourceNotFound = errors.New("repo: source not found")

type SourceRepo struct {
	pool *pgxpool.Pool
}

func NewSourceRepo(pool *pgxpool.Pool) *SourceRepo {
	return &SourceRepo{pool: pool}
}

func (r *SourceRepo) Create(ctx context.Context, s *domain.Source) (*domain.Source, error) {
	var out domain.Source
	err := r.pool.QueryRow(ctx,
		`INSERT INTO sources (project_id, type, name, url, config)
		 VALUES ($1, $2, $3, $4, $5)
		 RETURNING id, project_id, type, name, url, config, status, last_fetched_at, last_error, created_at, updated_at`,
		s.ProjectID, s.Type, s.Name, s.URL, s.Config,
	).Scan(&out.ID, &out.ProjectID, &out.Type, &out.Name, &out.URL, &out.Config,
		&out.Status, &out.LastFetchedAt, &out.LastError, &out.CreatedAt, &out.UpdatedAt)
	if err != nil {
		return nil, fmt.Errorf("repo: create source: %w", err)
	}
	return &out, nil
}

func (r *SourceRepo) ListByProject(ctx context.Context, projectID uuid.UUID) ([]domain.Source, error) {
	rows, err := r.pool.Query(ctx,
		`SELECT id, project_id, type, name, url, config, status, last_fetched_at, last_error, created_at, updated_at
		 FROM sources WHERE project_id = $1 ORDER BY created_at`,
		projectID,
	)
	if err != nil {
		return nil, fmt.Errorf("repo: list sources: %w", err)
	}
	defer rows.Close()

	var sources []domain.Source
	for rows.Next() {
		var s domain.Source
		if err := rows.Scan(&s.ID, &s.ProjectID, &s.Type, &s.Name, &s.URL, &s.Config,
			&s.Status, &s.LastFetchedAt, &s.LastError, &s.CreatedAt, &s.UpdatedAt); err != nil {
			return nil, fmt.Errorf("repo: scan source: %w", err)
		}
		sources = append(sources, s)
	}
	return sources, rows.Err()
}

func (r *SourceRepo) GetByID(ctx context.Context, id uuid.UUID) (*domain.Source, error) {
	var s domain.Source
	err := r.pool.QueryRow(ctx,
		`SELECT id, project_id, type, name, url, config, status, last_fetched_at, last_error, created_at, updated_at
		 FROM sources WHERE id = $1`,
		id,
	).Scan(&s.ID, &s.ProjectID, &s.Type, &s.Name, &s.URL, &s.Config,
		&s.Status, &s.LastFetchedAt, &s.LastError, &s.CreatedAt, &s.UpdatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrSourceNotFound
		}
		return nil, fmt.Errorf("repo: get source: %w", err)
	}
	return &s, nil
}

func (r *SourceRepo) Update(ctx context.Context, id uuid.UUID, name, url string, config json.RawMessage) (*domain.Source, error) {
	var out domain.Source
	err := r.pool.QueryRow(ctx,
		`UPDATE sources SET name = $2, url = $3, config = $4, updated_at = NOW()
		 WHERE id = $1
		 RETURNING id, project_id, type, name, url, config, status, last_fetched_at, last_error, created_at, updated_at`,
		id, name, url, config,
	).Scan(&out.ID, &out.ProjectID, &out.Type, &out.Name, &out.URL, &out.Config,
		&out.Status, &out.LastFetchedAt, &out.LastError, &out.CreatedAt, &out.UpdatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrSourceNotFound
		}
		return nil, fmt.Errorf("repo: update source: %w", err)
	}
	return &out, nil
}

func (r *SourceRepo) Toggle(ctx context.Context, id uuid.UUID) (*domain.Source, error) {
	var out domain.Source
	err := r.pool.QueryRow(ctx,
		`UPDATE sources SET
			status = CASE WHEN status = 'active' THEN 'disabled' ELSE 'active' END,
			updated_at = NOW()
		 WHERE id = $1
		 RETURNING id, project_id, type, name, url, config, status, last_fetched_at, last_error, created_at, updated_at`,
		id,
	).Scan(&out.ID, &out.ProjectID, &out.Type, &out.Name, &out.URL, &out.Config,
		&out.Status, &out.LastFetchedAt, &out.LastError, &out.CreatedAt, &out.UpdatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrSourceNotFound
		}
		return nil, fmt.Errorf("repo: toggle source: %w", err)
	}
	return &out, nil
}

func (r *SourceRepo) Delete(ctx context.Context, id uuid.UUID) error {
	tag, err := r.pool.Exec(ctx, `DELETE FROM sources WHERE id = $1`, id)
	if err != nil {
		return fmt.Errorf("repo: delete source: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return ErrSourceNotFound
	}
	return nil
}
