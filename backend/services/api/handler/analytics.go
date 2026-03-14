package handler

import (
	"net/http"
	"time"

	"github.com/brandradar/pkg/httputil"
	"github.com/brandradar/services/api/repo"
	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

type AnalyticsHandler struct {
	pool     *pgxpool.Pool
	mentions *repo.MentionRepo
}

func NewAnalyticsHandler(pool *pgxpool.Pool, mentions *repo.MentionRepo) *AnalyticsHandler {
	return &AnalyticsHandler{pool: pool, mentions: mentions}
}

type AnalyticsSummary struct {
	TotalMentions    int64                `json:"total_mentions"`
	SentimentCounts  map[string]int64     `json:"sentiment_counts"`
	TopSources       []SourceMentionCount `json:"top_sources"`
}

type SourceMentionCount struct {
	SourceID uuid.UUID `json:"source_id"`
	Count    int64     `json:"count"`
}

// Summary godoc
// @Summary Analytics summary
// @Tags analytics
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Success 200 {object} AnalyticsSummary
// @Router /projects/{projectID}/analytics [get]
func (h *AnalyticsHandler) Summary(w http.ResponseWriter, r *http.Request) {
	projectID, err := uuid.Parse(chi.URLParam(r, "projectID"))
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid project id")
		return
	}

	total, err := h.mentions.Count(r.Context(), projectID)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "failed to count mentions")
		return
	}

	sentiments, err := h.sentimentCounts(r, projectID)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "failed to get sentiment counts")
		return
	}

	topSources, err := h.topSources(r, projectID)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "failed to get top sources")
		return
	}

	httputil.JSON(w, http.StatusOK, AnalyticsSummary{
		TotalMentions:   total,
		SentimentCounts: sentiments,
		TopSources:      topSources,
	})
}

// Sentiment godoc
// @Summary Sentiment distribution
// @Tags analytics
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Success 200 {object} map[string]int64
// @Router /projects/{projectID}/analytics/sentiment [get]
func (h *AnalyticsHandler) Sentiment(w http.ResponseWriter, r *http.Request) {
	projectID, err := uuid.Parse(chi.URLParam(r, "projectID"))
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid project id")
		return
	}

	counts, err := h.sentimentCounts(r, projectID)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "failed to get sentiment counts")
		return
	}

	httputil.JSON(w, http.StatusOK, counts)
}

// Sources godoc
// @Summary Top sources by mention count
// @Tags analytics
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Success 200 {array} SourceMentionCount
// @Router /projects/{projectID}/analytics/sources [get]
func (h *AnalyticsHandler) Sources(w http.ResponseWriter, r *http.Request) {
	projectID, err := uuid.Parse(chi.URLParam(r, "projectID"))
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid project id")
		return
	}

	sources, err := h.topSources(r, projectID)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "failed to get top sources")
		return
	}

	if sources == nil {
		sources = []SourceMentionCount{}
	}

	httputil.JSON(w, http.StatusOK, sources)
}

func (h *AnalyticsHandler) sentimentCounts(r *http.Request, projectID uuid.UUID) (map[string]int64, error) {
	rows, err := h.pool.Query(r.Context(),
		`SELECT COALESCE(sentiment_label, 'unknown'), COUNT(*) FROM mentions WHERE project_id = $1 GROUP BY sentiment_label`, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	counts := make(map[string]int64)
	for rows.Next() {
		var sentiment string
		var count int64
		if err := rows.Scan(&sentiment, &count); err != nil {
			return nil, err
		}
		counts[sentiment] = count
	}
	return counts, rows.Err()
}

func (h *AnalyticsHandler) topSources(r *http.Request, projectID uuid.UUID) ([]SourceMentionCount, error) {
	rows, err := h.pool.Query(r.Context(),
		`SELECT source_id, COUNT(*) as cnt FROM mentions WHERE project_id = $1
		 GROUP BY source_id ORDER BY cnt DESC LIMIT 10`, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var sources []SourceMentionCount
	for rows.Next() {
		var s SourceMentionCount
		if err := rows.Scan(&s.SourceID, &s.Count); err != nil {
			return nil, err
		}
		sources = append(sources, s)
	}
	return sources, rows.Err()
}

// Timeline godoc
// @Summary Mentions timeline
// @Tags analytics
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param since query string false "Since (RFC3339)"
// @Param until query string false "Until (RFC3339)"
// @Param interval query string false "Interval (hour/day)" default(day)
// @Success 200 {array} repo.TimelinePoint
// @Router /projects/{projectID}/analytics/timeline [get]
func (h *AnalyticsHandler) Timeline(w http.ResponseWriter, r *http.Request) {
	projectID, err := uuid.Parse(chi.URLParam(r, "projectID"))
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid project id")
		return
	}

	since := time.Now().AddDate(0, 0, -7)
	if v := r.URL.Query().Get("since"); v != "" {
		if t, err := time.Parse(time.RFC3339, v); err == nil {
			since = t
		}
	}

	until := time.Now()
	if v := r.URL.Query().Get("until"); v != "" {
		if t, err := time.Parse(time.RFC3339, v); err == nil {
			until = t
		}
	}

	interval := r.URL.Query().Get("interval")
	if interval == "" {
		interval = "day"
	}

	points, err := h.mentions.Timeline(r.Context(), projectID, since, until, interval)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "failed to get timeline")
		return
	}

	if points == nil {
		points = []repo.TimelinePoint{}
	}

	httputil.JSON(w, http.StatusOK, points)
}
