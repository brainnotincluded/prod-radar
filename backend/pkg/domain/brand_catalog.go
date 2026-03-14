package domain

import "time"

type BrandCatalog struct {
	ID        int       `json:"id" db:"id"`
	Slug      string    `json:"slug" db:"slug"`
	Name      string    `json:"name" db:"name"`
	SourceURL string    `json:"source_url" db:"source_url"`
	CreatedAt time.Time `json:"created_at" db:"created_at"`
}
