package repo

import (
	"context"
	"fmt"

	"github.com/brandradar/pkg/domain"
	"github.com/jackc/pgx/v5/pgxpool"
)

type BrandCatalogRepo struct {
	pool *pgxpool.Pool
}

func NewBrandCatalogRepo(pool *pgxpool.Pool) *BrandCatalogRepo {
	return &BrandCatalogRepo{pool: pool}
}

func (r *BrandCatalogRepo) Search(ctx context.Context, query string, limit int) ([]domain.BrandCatalog, error) {
	rows, err := r.pool.Query(ctx,
		`SELECT id, slug, name, source_url, created_at
		 FROM brand_catalog
		 WHERE name ILIKE '%' || $1 || '%'
		 ORDER BY similarity(name, $1) DESC
		 LIMIT $2`,
		query, limit,
	)
	if err != nil {
		return nil, fmt.Errorf("repo: search brand catalog: %w", err)
	}
	defer rows.Close()

	var brands []domain.BrandCatalog
	for rows.Next() {
		var b domain.BrandCatalog
		if err := rows.Scan(&b.ID, &b.Slug, &b.Name, &b.SourceURL, &b.CreatedAt); err != nil {
			return nil, fmt.Errorf("repo: scan brand catalog: %w", err)
		}
		brands = append(brands, b)
	}
	return brands, rows.Err()
}

func (r *BrandCatalogRepo) List(ctx context.Context, offset, limit int) ([]domain.BrandCatalog, int, error) {
	var total int
	err := r.pool.QueryRow(ctx, `SELECT COUNT(*) FROM brand_catalog`).Scan(&total)
	if err != nil {
		return nil, 0, fmt.Errorf("repo: count brand catalog: %w", err)
	}

	rows, err := r.pool.Query(ctx,
		`SELECT id, slug, name, source_url, created_at
		 FROM brand_catalog ORDER BY name LIMIT $1 OFFSET $2`,
		limit, offset,
	)
	if err != nil {
		return nil, 0, fmt.Errorf("repo: list brand catalog: %w", err)
	}
	defer rows.Close()

	var brands []domain.BrandCatalog
	for rows.Next() {
		var b domain.BrandCatalog
		if err := rows.Scan(&b.ID, &b.Slug, &b.Name, &b.SourceURL, &b.CreatedAt); err != nil {
			return nil, 0, fmt.Errorf("repo: scan brand catalog: %w", err)
		}
		brands = append(brands, b)
	}
	return brands, total, rows.Err()
}
