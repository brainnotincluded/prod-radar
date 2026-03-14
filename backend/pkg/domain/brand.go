package domain

import (
	"time"

	"github.com/google/uuid"
)

var DefaultRiskWords = []string{
	"утечка",
	"слив",
	"взлом",
	"мошенничество",
	"сбой",
	"скандал",
	"штраф",
	"уязвимость",
	"фишинг",
	"отзыв продукции",
	"судебный иск",
	"банкротство",
	"увольнение",
	"задержание",
	"расследование",
}

type Brand struct {
	ID        uuid.UUID `json:"id" db:"id"`
	ProjectID uuid.UUID `json:"project_id" db:"project_id"`
	CatalogID *int      `json:"catalog_id,omitempty" db:"catalog_id"`
	Name      string    `json:"name" db:"name"`
	Keywords  []string  `json:"keywords" db:"keywords"`
	Exclusions []string `json:"exclusions" db:"exclusions"`
	RiskWords []string  `json:"risk_words" db:"risk_words"`
	CreatedAt time.Time `json:"created_at" db:"created_at"`
	UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}
