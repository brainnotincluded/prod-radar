package handler

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"

	"github.com/brandradar/pkg/auth"
	"github.com/brandradar/pkg/domain"
	"github.com/brandradar/pkg/httputil"
	"github.com/brandradar/pkg/messaging"
	"github.com/brandradar/services/api/repo"
	"github.com/nats-io/nats.go/jetstream"
)

type SourceHandler struct {
	sources  *repo.SourceRepo
	projects *repo.ProjectRepo
	js       jetstream.JetStream
}

func NewSourceHandler(sources *repo.SourceRepo, projects *repo.ProjectRepo, js jetstream.JetStream) *SourceHandler {
	return &SourceHandler{sources: sources, projects: projects, js: js}
}

type createSourceRequest struct {
	Type   string          `json:"type" validate:"required,oneof=rss web telegram"`
	Name   string          `json:"name" validate:"required,min=1,max=255"`
	URL    string          `json:"url" validate:"required,url"`
	Config json.RawMessage `json:"config"`
}

// Create godoc
// @Summary Create a source
// @Tags sources
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param body body createSourceRequest true "Source data"
// @Success 201 {object} domain.Source
// @Failure 422 {object} httputil.ErrorResponse
// @Router /projects/{projectID}/sources [post]
func (h *SourceHandler) Create(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.UserIDFromContext(r.Context())
	if !ok {
		httputil.Error(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	projectID, err := httputil.URLParamUUID(r, "projectID")
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, err.Error())
		return
	}

	project, err := h.projects.GetByID(r.Context(), projectID)
	if err != nil {
		if errors.Is(err, repo.ErrProjectNotFound) {
			httputil.Error(w, http.StatusNotFound, "project not found")
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}
	if project.UserID != userID {
		httputil.Error(w, http.StatusForbidden, "forbidden")
		return
	}

	var req createSourceRequest
	if err := httputil.DecodeJSON(r, &req); err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if err := validate.Struct(req); err != nil {
		httputil.ErrorWithDetails(w, http.StatusUnprocessableEntity, "validation failed", err.Error())
		return
	}

	if req.Config == nil {
		req.Config = json.RawMessage(`{}`)
	}

	src := &domain.Source{
		ProjectID: projectID,
		Type:      domain.SourceType(req.Type),
		Name:      req.Name,
		URL:       req.URL,
		Config:    req.Config,
	}

	created, err := h.sources.Create(r.Context(), src)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	h.publishSourcesChanged(r)
	httputil.JSON(w, http.StatusCreated, created)
}

// List godoc
// @Summary List project sources
// @Tags sources
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Success 200 {array} domain.Source
// @Router /projects/{projectID}/sources [get]
func (h *SourceHandler) List(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.UserIDFromContext(r.Context())
	if !ok {
		httputil.Error(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	projectID, err := httputil.URLParamUUID(r, "projectID")
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, err.Error())
		return
	}

	project, err := h.projects.GetByID(r.Context(), projectID)
	if err != nil {
		if errors.Is(err, repo.ErrProjectNotFound) {
			httputil.Error(w, http.StatusNotFound, "project not found")
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}
	if project.UserID != userID {
		httputil.Error(w, http.StatusForbidden, "forbidden")
		return
	}

	sources, err := h.sources.ListByProject(r.Context(), projectID)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}
	if sources == nil {
		sources = []domain.Source{}
	}

	httputil.JSON(w, http.StatusOK, sources)
}

type updateSourceRequest struct {
	Name   string          `json:"name" validate:"required,min=1,max=255"`
	URL    string          `json:"url" validate:"required,url"`
	Config json.RawMessage `json:"config"`
}

// Update godoc
// @Summary Update a source
// @Tags sources
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param sourceID path string true "Source ID"
// @Param body body updateSourceRequest true "Source data"
// @Success 200 {object} domain.Source
// @Failure 404 {object} httputil.ErrorResponse
// @Router /projects/{projectID}/sources/{sourceID} [put]
func (h *SourceHandler) Update(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.UserIDFromContext(r.Context())
	if !ok {
		httputil.Error(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	sourceID, err := httputil.URLParamUUID(r, "sourceID")
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, err.Error())
		return
	}

	src, err := h.sources.GetByID(r.Context(), sourceID)
	if err != nil {
		if errors.Is(err, repo.ErrSourceNotFound) {
			httputil.Error(w, http.StatusNotFound, "source not found")
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	project, err := h.projects.GetByID(r.Context(), src.ProjectID)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}
	if project.UserID != userID {
		httputil.Error(w, http.StatusForbidden, "forbidden")
		return
	}

	var req updateSourceRequest
	if err := httputil.DecodeJSON(r, &req); err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if err := validate.Struct(req); err != nil {
		httputil.ErrorWithDetails(w, http.StatusUnprocessableEntity, "validation failed", err.Error())
		return
	}
	if req.Config == nil {
		req.Config = json.RawMessage(`{}`)
	}

	updated, err := h.sources.Update(r.Context(), sourceID, req.Name, req.URL, req.Config)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	h.publishSourcesChanged(r)
	httputil.JSON(w, http.StatusOK, updated)
}

// Toggle godoc
// @Summary Toggle source status
// @Tags sources
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param sourceID path string true "Source ID"
// @Success 200 {object} domain.Source
// @Failure 404 {object} httputil.ErrorResponse
// @Router /projects/{projectID}/sources/{sourceID}/toggle [post]
func (h *SourceHandler) Toggle(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.UserIDFromContext(r.Context())
	if !ok {
		httputil.Error(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	sourceID, err := httputil.URLParamUUID(r, "sourceID")
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, err.Error())
		return
	}

	src, err := h.sources.GetByID(r.Context(), sourceID)
	if err != nil {
		if errors.Is(err, repo.ErrSourceNotFound) {
			httputil.Error(w, http.StatusNotFound, "source not found")
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	project, err := h.projects.GetByID(r.Context(), src.ProjectID)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}
	if project.UserID != userID {
		httputil.Error(w, http.StatusForbidden, "forbidden")
		return
	}

	toggled, err := h.sources.Toggle(r.Context(), sourceID)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	h.publishSourcesChanged(r)
	httputil.JSON(w, http.StatusOK, toggled)
}

// Delete godoc
// @Summary Delete a source
// @Tags sources
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param sourceID path string true "Source ID"
// @Success 204
// @Failure 404 {object} httputil.ErrorResponse
// @Router /projects/{projectID}/sources/{sourceID} [delete]
func (h *SourceHandler) Delete(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.UserIDFromContext(r.Context())
	if !ok {
		httputil.Error(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	sourceID, err := httputil.URLParamUUID(r, "sourceID")
	if err != nil {
		httputil.Error(w, http.StatusBadRequest, err.Error())
		return
	}

	src, err := h.sources.GetByID(r.Context(), sourceID)
	if err != nil {
		if errors.Is(err, repo.ErrSourceNotFound) {
			httputil.Error(w, http.StatusNotFound, "source not found")
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	project, err := h.projects.GetByID(r.Context(), src.ProjectID)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}
	if project.UserID != userID {
		httputil.Error(w, http.StatusForbidden, "forbidden")
		return
	}

	if err := h.sources.Delete(r.Context(), sourceID); err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	h.publishSourcesChanged(r)
	w.WriteHeader(http.StatusNoContent)
}

func (h *SourceHandler) publishSourcesChanged(r *http.Request) {
	if h.js == nil {
		return
	}
	pub := messaging.NewPublisher(h.js)
	if err := pub.Publish(r.Context(), messaging.SubjectSourcesChanged, map[string]string{"event": "sources_changed"}); err != nil {
		slog.Warn("publish sources.changed", "error", err)
	}
}
