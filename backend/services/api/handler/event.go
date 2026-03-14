package handler

import (
	"net/http"
	"strconv"

	"github.com/brandradar/pkg/httputil"
	"github.com/brandradar/services/api/repo"
	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
)

type EventHandler struct {
	events *repo.EventRepo
}

func NewEventHandler(events *repo.EventRepo) *EventHandler {
	return &EventHandler{events: events}
}

// List godoc
// @Summary List project events
// @Tags events
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param limit query int false "Limit" default(50)
// @Param offset query int false "Offset" default(0)
// @Success 200 {array} domain.Event
// @Router /projects/{projectID}/events [get]
func (h *EventHandler) List(w http.ResponseWriter, r *http.Request) {
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

	events, err := h.events.ListByProject(r.Context(), projectID, limit, offset)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "failed to list events")
		return
	}

	if events == nil {
		events = []repo.EventRow{}
	}

	httputil.JSON(w, http.StatusOK, events)
}
