package domain

import (
	"time"

	"github.com/google/uuid"
)

type MentionStatus string

const (
	MentionStatusPendingML MentionStatus = "pending_ml"
	MentionStatusEnriched  MentionStatus = "enriched"
	MentionStatusReady     MentionStatus = "ready"
	MentionStatusDismissed MentionStatus = "dismissed"
)

type Mention struct {
	ID               uuid.UUID     `json:"id" db:"id"`
	ProjectID        uuid.UUID     `json:"project_id" db:"project_id"`
	SourceID         uuid.UUID     `json:"source_id" db:"source_id"`
	ExternalID       string        `json:"external_id" db:"external_id"`
	Title            *string       `json:"title,omitempty" db:"title"`
	Text             string        `json:"text" db:"text"`
	URL              *string       `json:"url,omitempty" db:"url"`
	PublishedAt      time.Time     `json:"published_at" db:"published_at"`
	FetchedAt        time.Time     `json:"fetched_at" db:"fetched_at"`
	MatchedKeywords  []string      `json:"matched_keywords" db:"matched_keywords"`
	MatchedRiskWords []string      `json:"matched_risk_words" db:"matched_risk_words"`
	SentimentLabel   *string       `json:"sentiment_label,omitempty" db:"sentiment_label"`
	SentimentScore   *float64      `json:"sentiment_score,omitempty" db:"sentiment_score"`
	Embedding        []float32     `json:"-" db:"embedding"`
	ClusterID        *uuid.UUID    `json:"cluster_id,omitempty" db:"cluster_id"`
	IsDuplicate      bool          `json:"is_duplicate" db:"is_duplicate"`
	Status           MentionStatus `json:"status" db:"status"`
	CreatedAt        time.Time     `json:"created_at" db:"created_at"`
}
