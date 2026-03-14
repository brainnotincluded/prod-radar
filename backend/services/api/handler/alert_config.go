package handler

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/brandradar/pkg/auth"
	"github.com/brandradar/pkg/httputil"
	"github.com/brandradar/services/api/repo"
)

type AlertConfigHandler struct {
	configs  *repo.AlertConfigRepo
	projects *repo.ProjectRepo
}

func NewAlertConfigHandler(configs *repo.AlertConfigRepo, projects *repo.ProjectRepo) *AlertConfigHandler {
	return &AlertConfigHandler{configs: configs, projects: projects}
}

type upsertAlertConfigRequest struct {
	SpikeThreshold       int             `json:"spike_threshold" validate:"required,min=1"`
	SpikeWindowMinutes   int             `json:"spike_window_minutes" validate:"required,min=1"`
	CooldownMinutes      int             `json:"cooldown_minutes" validate:"required,min=1"`
	NotificationChannels json.RawMessage `json:"notification_channels"`
	Enabled              *bool           `json:"enabled" validate:"required"`
}

// Get godoc
// @Summary Get alert configuration
// @Tags alert-config
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Success 200 {object} domain.AlertConfig
// @Router /projects/{projectID}/alert-config [get]
func (h *AlertConfigHandler) Get(w http.ResponseWriter, r *http.Request) {
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

	config, err := h.configs.GetByProject(r.Context(), projectID)
	if err != nil {
		if errors.Is(err, repo.ErrAlertConfigNotFound) {
			httputil.JSON(w, http.StatusOK, map[string]interface{}{
				"spike_threshold":       50,
				"spike_window_minutes":  60,
				"cooldown_minutes":      30,
				"notification_channels": []interface{}{},
				"enabled":               true,
			})
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	httputil.JSON(w, http.StatusOK, config)
}

// Upsert godoc
// @Summary Create or update alert configuration
// @Tags alert-config
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param body body upsertAlertConfigRequest true "Alert config data"
// @Success 200 {object} domain.AlertConfig
// @Failure 422 {object} httputil.ErrorResponse
// @Router /projects/{projectID}/alert-config [put]
func (h *AlertConfigHandler) Upsert(w http.ResponseWriter, r *http.Request) {
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

	var req upsertAlertConfigRequest
	if err := httputil.DecodeJSON(r, &req); err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if err := validate.Struct(req); err != nil {
		httputil.ErrorWithDetails(w, http.StatusUnprocessableEntity, "validation failed", err.Error())
		return
	}
	if req.NotificationChannels == nil {
		req.NotificationChannels = json.RawMessage(`[]`)
	}

	config, err := h.configs.Upsert(r.Context(), projectID,
		req.SpikeThreshold, req.SpikeWindowMinutes, req.CooldownMinutes,
		req.NotificationChannels, *req.Enabled)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	httputil.JSON(w, http.StatusOK, config)
}
