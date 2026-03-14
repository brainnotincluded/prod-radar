package repo

import (
	"context"
	"errors"
	"fmt"

	"github.com/brandradar/pkg/domain"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

var ErrProjectNotFound = errors.New("repo: project not found")

type ProjectRepo struct {
	pool *pgxpool.Pool
}

func NewProjectRepo(pool *pgxpool.Pool) *ProjectRepo {
	return &ProjectRepo{pool: pool}
}

func (r *ProjectRepo) Create(ctx context.Context, userID uuid.UUID, name string, description *string) (*domain.Project, error) {
	var p domain.Project
	err := r.pool.QueryRow(ctx,
		`INSERT INTO projects (user_id, name, description) VALUES ($1, $2, $3)
		 RETURNING id, user_id, name, description, created_at, updated_at`,
		userID, name, description,
	).Scan(&p.ID, &p.UserID, &p.Name, &p.Description, &p.CreatedAt, &p.UpdatedAt)
	if err != nil {
		return nil, fmt.Errorf("repo: create project: %w", err)
	}
	return &p, nil
}

func (r *ProjectRepo) ListByUser(ctx context.Context, userID uuid.UUID) ([]domain.Project, error) {
	rows, err := r.pool.Query(ctx,
		`SELECT id, user_id, name, description, created_at, updated_at
		 FROM projects WHERE user_id = $1 ORDER BY created_at DESC`,
		userID,
	)
	if err != nil {
		return nil, fmt.Errorf("repo: list projects: %w", err)
	}
	defer rows.Close()

	var projects []domain.Project
	for rows.Next() {
		var p domain.Project
		if err := rows.Scan(&p.ID, &p.UserID, &p.Name, &p.Description, &p.CreatedAt, &p.UpdatedAt); err != nil {
			return nil, fmt.Errorf("repo: scan project: %w", err)
		}
		projects = append(projects, p)
	}
	return projects, rows.Err()
}

func (r *ProjectRepo) GetByID(ctx context.Context, id uuid.UUID) (*domain.Project, error) {
	var p domain.Project
	err := r.pool.QueryRow(ctx,
		`SELECT id, user_id, name, description, created_at, updated_at FROM projects WHERE id = $1`,
		id,
	).Scan(&p.ID, &p.UserID, &p.Name, &p.Description, &p.CreatedAt, &p.UpdatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrProjectNotFound
		}
		return nil, fmt.Errorf("repo: get project: %w", err)
	}
	return &p, nil
}

func (r *ProjectRepo) Update(ctx context.Context, id uuid.UUID, name string, description *string) (*domain.Project, error) {
	var p domain.Project
	err := r.pool.QueryRow(ctx,
		`UPDATE projects SET name = $2, description = $3, updated_at = NOW()
		 WHERE id = $1
		 RETURNING id, user_id, name, description, created_at, updated_at`,
		id, name, description,
	).Scan(&p.ID, &p.UserID, &p.Name, &p.Description, &p.CreatedAt, &p.UpdatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrProjectNotFound
		}
		return nil, fmt.Errorf("repo: update project: %w", err)
	}
	return &p, nil
}

func (r *ProjectRepo) Delete(ctx context.Context, id uuid.UUID) error {
	tag, err := r.pool.Exec(ctx, `DELETE FROM projects WHERE id = $1`, id)
	if err != nil {
		return fmt.Errorf("repo: delete project: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return ErrProjectNotFound
	}
	return nil
}
