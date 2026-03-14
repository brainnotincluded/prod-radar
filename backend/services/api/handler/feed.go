package handler

import (
	"net/http"
	"strconv"
	"time"

	"github.com/brandradar/pkg/httputil"
	"github.com/brandradar/services/api/repo"
	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
)

type FeedHandler struct {
	mentions *repo.MentionRepo
}

func NewFeedHandler(mentions *repo.MentionRepo) *FeedHandler {
	return &FeedHandler{mentions: mentions}
}

// List godoc
// @Summary List mentions feed
// @Tags feed
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param sentiment query string false "Filter by sentiment"
// @Param status query string false "Filter by status"
// @Param since query string false "Filter since (RFC3339)"
// @Param until query string false "Filter until (RFC3339)"
// @Param brand_id query string false "Filter by brand ID"
// @Param source_id query string false "Filter by source ID"
// @Param search query string false "Text search"
// @Param has_risk query bool false "Filter by risk words"
// @Param limit query int false "Limit" default(50)
// @Param offset query int false "Offset" default(0)
// @Success 200 {array} repo.MentionRow
// @Router /projects/{projectID}/feed [get]
func (h *FeedHandler) List(w http.ResponseWriter, r *http.Request) {
	projectID, err := uuid.Parse(chi.URLParam(r, "projectID"))
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid project id")
		return
	}

	limit := 50
	if v := r.URL.Query().Get("limit"); v != "" {
		if l, err := strconv.Atoi(v); err == nil && l > 0 && l <= 200 {
			limit = l
		}
	}

	offset := 0
	if v := r.URL.Query().Get("offset"); v != "" {
		if o, err := strconv.Atoi(v); err == nil && o >= 0 {
			offset = o
		}
	}

	f := repo.MentionFilter{
		ProjectID: projectID,
		Limit:     limit,
		Offset:    offset,
	}

	if v := r.URL.Query().Get("sentiment"); v != "" {
		f.Sentiment = &v
	}
	if v := r.URL.Query().Get("status"); v != "" {
		f.Status = &v
	}
	if v := r.URL.Query().Get("since"); v != "" {
		if t, err := time.Parse(time.RFC3339, v); err == nil {
			f.Since = &t
		}
	}
	if v := r.URL.Query().Get("until"); v != "" {
		if t, err := time.Parse(time.RFC3339, v); err == nil {
			f.Until = &t
		}
	}
	if v := r.URL.Query().Get("brand_id"); v != "" {
		if id, err := uuid.Parse(v); err == nil {
			f.BrandID = &id
		}
	}
	if v := r.URL.Query().Get("source_id"); v != "" {
		if id, err := uuid.Parse(v); err == nil {
			f.SourceID = &id
		}
	}
	if v := r.URL.Query().Get("search"); v != "" {
		f.Search = &v
	}
	if r.URL.Query().Get("has_risk") == "true" {
		hr := true
		f.HasRisk = &hr
	}

	mentions, err := h.mentions.List(r.Context(), f)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "failed to list mentions")
		return
	}

	if mentions == nil {
		mentions = []repo.MentionRow{}
	}

	httputil.JSON(w, http.StatusOK, mentions)
}

// Get godoc
// @Summary Get mention details
// @Tags feed
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param mentionID path string true "Mention ID"
// @Success 200 {object} repo.MentionRow
// @Failure 404 {object} httputil.ErrorResponse
// @Router /projects/{projectID}/feed/{mentionID} [get]
func (h *FeedHandler) Get(w http.ResponseWriter, r *http.Request) {
	projectID, err := uuid.Parse(chi.URLParam(r, "projectID"))
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid project id")
		return
	}

	mentionID, err := uuid.Parse(chi.URLParam(r, "mentionID"))
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid mention id")
		return
	}

	mention, err := h.mentions.GetByID(r.Context(), projectID, mentionID)
	if err != nil {
		httputil.Error(w, http.StatusNotFound, "mention not found")
		return
	}

	httputil.JSON(w, http.StatusOK, mention)
}

// Duplicates godoc
// @Summary List mention duplicates
// @Tags feed
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param mentionID path string true "Mention ID"
// @Success 200 {array} repo.MentionRow
// @Failure 404 {object} httputil.ErrorResponse
// @Router /projects/{projectID}/feed/{mentionID}/duplicates [get]
func (h *FeedHandler) Duplicates(w http.ResponseWriter, r *http.Request) {
	projectID, err := uuid.Parse(chi.URLParam(r, "projectID"))
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid project id")
		return
	}

	mentionID, err := uuid.Parse(chi.URLParam(r, "mentionID"))
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid mention id")
		return
	}

	duplicates, err := h.mentions.Duplicates(r.Context(), projectID, mentionID)
	if err != nil {
		httputil.Error(w, http.StatusNotFound, "mention not found")
		return
	}

	if duplicates == nil {
		duplicates = []repo.MentionRow{}
	}

	httputil.JSON(w, http.StatusOK, duplicates)
}

// Dismiss godoc
// @Summary Dismiss a mention
// @Tags feed
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param mentionID path string true "Mention ID"
// @Success 200 {object} map[string]string
// @Router /projects/{projectID}/feed/{mentionID}/dismiss [post]
func (h *FeedHandler) Dismiss(w http.ResponseWriter, r *http.Request) {
	projectID, err := uuid.Parse(chi.URLParam(r, "projectID"))
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid project id")
		return
	}

	mentionID, err := uuid.Parse(chi.URLParam(r, "mentionID"))
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid mention id")
		return
	}

	if err := h.mentions.Dismiss(r.Context(), projectID, mentionID); err != nil {
		httputil.Error(w, http.StatusInternalServerError, "failed to dismiss mention")
		return
	}

	httputil.JSON(w, http.StatusOK, map[string]string{"status": "dismissed"})
}
