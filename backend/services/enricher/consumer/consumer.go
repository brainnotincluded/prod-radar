package consumer

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/brandradar/pkg/domain"
	"github.com/brandradar/pkg/messaging"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/nats-io/nats.go/jetstream"
)

type FilteredMention struct {
	MentionID        uuid.UUID `json:"mention_id"`
	SourceID         uuid.UUID `json:"source_id"`
	BrandID          uuid.UUID `json:"brand_id"`
	ProjectID        uuid.UUID `json:"project_id"`
	ExternalID       string    `json:"external_id"`
	URL              string    `json:"url"`
	Title            string    `json:"title"`
	Content          string    `json:"content"`
	Author           string    `json:"author,omitempty"`
	PublishedAt      time.Time `json:"published_at"`
	MatchedKeywords  []string  `json:"matched_keywords"`
	MatchedRiskWords []string  `json:"matched_risk_words"`
}

type mlRequest struct {
	Text string `json:"text"`
}

type mlResponse struct {
	SentimentLabel string    `json:"sentiment_label"`
	SentimentScore float64   `json:"sentiment_score"`
	Embedding      []float32 `json:"embedding"`
}

type Consumer struct {
	pool     *pgxpool.Pool
	js       jetstream.JetStream
	mlURL    string
	mlClient *http.Client
}

func New(pool *pgxpool.Pool, js jetstream.JetStream, mlURL string) *Consumer {
	return &Consumer{
		pool:     pool,
		js:       js,
		mlURL:    mlURL,
		mlClient: &http.Client{Timeout: 30 * time.Second},
	}
}

func (c *Consumer) Run(ctx context.Context) error {
	cons := messaging.NewConsumer(c.js, messaging.ConsumerConfig{
		Stream:        messaging.StreamMentions,
		Subjects:      []string{"mentions.>"},
		Durable:       "enricher",
		FilterSubject: messaging.SubjectMentionsFiltered,
	})

	return cons.Run(ctx, func(ctx context.Context, data []byte) error {
		return c.handle(ctx, data)
	})
}

func (c *Consumer) handle(ctx context.Context, data []byte) error {
	var fm FilteredMention
	if err := json.Unmarshal(data, &fm); err != nil {
		slog.Warn("unmarshal mention", "error", err)
		return nil
	}

	ml, err := c.callMLService(ctx, fm.Content)
	if err != nil {
		slog.Warn("ml service call failed, using defaults", "error", err)
		ml = &mlResponse{SentimentLabel: "neutral", SentimentScore: 0}
	}

	isDup := false
	if len(ml.Embedding) > 0 {
		isDup, err = c.checkVectorDedup(ctx, fm.ProjectID, ml.Embedding)
		if err != nil {
			slog.Warn("vector dedup check", "error", err)
		}
	}

	var clusterID *uuid.UUID
	if len(ml.Embedding) > 0 {
		clusterID = c.assignCluster(ctx, fm.ProjectID, ml.Embedding)
	}

	err = c.updateMention(ctx, fm.MentionID, ml, isDup, clusterID)
	if err != nil {
		return fmt.Errorf("update mention: %w", err)
	}

	readyMsg := struct {
		MentionID      uuid.UUID  `json:"mention_id"`
		ProjectID      uuid.UUID  `json:"project_id"`
		SentimentLabel string     `json:"sentiment_label"`
		SentimentScore float64    `json:"sentiment_score"`
		IsDuplicate    bool       `json:"is_duplicate"`
		ClusterID      *uuid.UUID `json:"cluster_id,omitempty"`
	}{
		MentionID:      fm.MentionID,
		ProjectID:      fm.ProjectID,
		SentimentLabel: ml.SentimentLabel,
		SentimentScore: ml.SentimentScore,
		IsDuplicate:    isDup,
		ClusterID:      clusterID,
	}

	pub := messaging.NewPublisher(c.js)
	if err := pub.Publish(ctx, messaging.SubjectMentionsReady, readyMsg); err != nil {
		return fmt.Errorf("publish mentions.ready: %w", err)
	}

	slog.Info("mention enriched",
		"mention_id", fm.MentionID,
		"sentiment", ml.SentimentLabel,
		"is_duplicate", isDup,
	)
	return nil
}

func (c *Consumer) callMLService(ctx context.Context, text string) (*mlResponse, error) {
	if c.mlURL == "" {
		return &mlResponse{SentimentLabel: "neutral", SentimentScore: 0}, nil
	}

	body, err := json.Marshal(mlRequest{Text: text})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.mlURL+"/analyze", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.mlClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ml service returned %d", resp.StatusCode)
	}

	var result mlResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *Consumer) checkVectorDedup(ctx context.Context, projectID uuid.UUID, embedding []float32) (bool, error) {
	embJSON, err := json.Marshal(embedding)
	if err != nil {
		return false, err
	}

	var exists bool
	err = c.pool.QueryRow(ctx,
		`SELECT EXISTS(
			SELECT 1 FROM mentions
			WHERE project_id = $1
			  AND embedding IS NOT NULL
			  AND status != $3
			  AND (embedding <=> $2::vector) < 0.05
		)`,
		projectID, string(embJSON), domain.MentionStatusDismissed,
	).Scan(&exists)
	return exists, err
}

func (c *Consumer) assignCluster(ctx context.Context, projectID uuid.UUID, embedding []float32) *uuid.UUID {
	embJSON, _ := json.Marshal(embedding)

	var clusterID uuid.UUID
	err := c.pool.QueryRow(ctx,
		`UPDATE mention_clusters SET mention_count = mention_count + 1, updated_at = NOW()
		 WHERE id = (
		   SELECT id FROM mention_clusters
		   WHERE project_id = $1
		     AND (centroid <=> $2::vector) < 0.15
		   ORDER BY centroid <=> $2::vector
		   LIMIT 1
		 )
		 RETURNING id`,
		projectID, string(embJSON),
	).Scan(&clusterID)

	if err == nil {
		return &clusterID
	}

	newID := uuid.New()
	_, err = c.pool.Exec(ctx,
		`INSERT INTO mention_clusters (id, project_id, centroid, mention_count)
		 VALUES ($1, $2, $3::vector, 1)`,
		newID, projectID, string(embJSON),
	)
	if err != nil {
		slog.Warn("create cluster", "error", err)
		return nil
	}
	return &newID
}

func (c *Consumer) updateMention(ctx context.Context, mentionID uuid.UUID, ml *mlResponse, isDup bool, clusterID *uuid.UUID) error {
	status := domain.MentionStatusEnriched
	if isDup {
		status = domain.MentionStatusDismissed
	}

	var embJSON *string
	if len(ml.Embedding) > 0 {
		b, _ := json.Marshal(ml.Embedding)
		s := string(b)
		embJSON = &s
	}

	_, err := c.pool.Exec(ctx,
		`UPDATE mentions SET
			sentiment_label = $2,
			sentiment_score = $3,
			embedding = $4::vector,
			is_duplicate = $5,
			cluster_id = $6,
			status = $7
		 WHERE id = $1`,
		mentionID, ml.SentimentLabel, ml.SentimentScore,
		embJSON, isDup, clusterID, status,
	)
	return err
}
