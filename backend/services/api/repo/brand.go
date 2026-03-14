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

var ErrBrandNotFound = errors.New("repo: brand not found")

type BrandRepo struct {
	pool *pgxpool.Pool
}

func NewBrandRepo(pool *pgxpool.Pool) *BrandRepo {
	return &BrandRepo{pool: pool}
}

func (r *BrandRepo) Create(ctx context.Context, projectID uuid.UUID, catalogID *int, name string, keywords, exclusions, riskWords []string) (*domain.Brand, error) {
	if len(riskWords) == 0 {
		riskWords = domain.DefaultRiskWords
	}
	var b domain.Brand
	err := r.pool.QueryRow(ctx,
		`INSERT INTO brands (project_id, catalog_id, name, keywords, exclusions, risk_words)
		 VALUES ($1, $2, $3, $4, $5, $6)
		 RETURNING id, project_id, catalog_id, name, keywords, exclusions, risk_words, created_at, updated_at`,
		projectID, catalogID, name, keywords, exclusions, riskWords,
	).Scan(&b.ID, &b.ProjectID, &b.CatalogID, &b.Name, &b.Keywords, &b.Exclusions, &b.RiskWords, &b.CreatedAt, &b.UpdatedAt)
	if err != nil {
		return nil, fmt.Errorf("repo: create brand: %w", err)
	}
	return &b, nil
}

func (r *BrandRepo) GetByProjectID(ctx context.Context, projectID uuid.UUID) (*domain.Brand, error) {
	var b domain.Brand
	err := r.pool.QueryRow(ctx,
		`SELECT id, project_id, catalog_id, name, keywords, exclusions, risk_words, created_at, updated_at
		 FROM brands WHERE project_id = $1`,
		projectID,
	).Scan(&b.ID, &b.ProjectID, &b.CatalogID, &b.Name, &b.Keywords, &b.Exclusions, &b.RiskWords, &b.CreatedAt, &b.UpdatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrBrandNotFound
		}
		return nil, fmt.Errorf("repo: get brand: %w", err)
	}
	return &b, nil
}

func (r *BrandRepo) Update(ctx context.Context, projectID uuid.UUID, catalogID *int, name string, keywords, exclusions, riskWords []string) (*domain.Brand, error) {
	var b domain.Brand
	err := r.pool.QueryRow(ctx,
		`UPDATE brands SET catalog_id = $2, name = $3, keywords = $4, exclusions = $5, risk_words = $6, updated_at = NOW()
		 WHERE project_id = $1
		 RETURNING id, project_id, catalog_id, name, keywords, exclusions, risk_words, created_at, updated_at`,
		projectID, catalogID, name, keywords, exclusions, riskWords,
	).Scan(&b.ID, &b.ProjectID, &b.CatalogID, &b.Name, &b.Keywords, &b.Exclusions, &b.RiskWords, &b.CreatedAt, &b.UpdatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrBrandNotFound
		}
		return nil, fmt.Errorf("repo: update brand: %w", err)
	}
	return &b, nil
}

func (r *BrandRepo) Delete(ctx context.Context, projectID uuid.UUID) error {
	tag, err := r.pool.Exec(ctx, `DELETE FROM brands WHERE project_id = $1`, projectID)
	if err != nil {
		return fmt.Errorf("repo: delete brand: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return ErrBrandNotFound
	}
	return nil
}
