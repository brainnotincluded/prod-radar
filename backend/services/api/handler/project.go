package handler

import (
	"errors"
	"net/http"

	"github.com/brandradar/pkg/auth"
	"github.com/brandradar/pkg/domain"
	"github.com/brandradar/pkg/httputil"
	"github.com/brandradar/services/api/repo"
)

type ProjectHandler struct {
	projects *repo.ProjectRepo
}

func NewProjectHandler(projects *repo.ProjectRepo) *ProjectHandler {
	return &ProjectHandler{projects: projects}
}

type createProjectRequest struct {
	Name        string  `json:"name" validate:"required,min=1,max=255"`
	Description *string `json:"description"`
}

type updateProjectRequest struct {
	Name        string  `json:"name" validate:"required,min=1,max=255"`
	Description *string `json:"description"`
}

// Create godoc
// @Summary Create a project
// @Tags projects
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param body body createProjectRequest true "Project data"
// @Success 201 {object} domain.Project
// @Failure 422 {object} httputil.ErrorResponse
// @Router /projects [post]
func (h *ProjectHandler) Create(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.UserIDFromContext(r.Context())
	if !ok {
		httputil.Error(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	var req createProjectRequest
	if err := httputil.DecodeJSON(r, &req); err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if err := validate.Struct(req); err != nil {
		httputil.ErrorWithDetails(w, http.StatusUnprocessableEntity, "validation failed", err.Error())
		return
	}

	project, err := h.projects.Create(r.Context(), userID, req.Name, req.Description)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	httputil.JSON(w, http.StatusCreated, project)
}

// List godoc
// @Summary List user projects
// @Tags projects
// @Produce json
// @Security BearerAuth
// @Success 200 {array} domain.Project
// @Router /projects [get]
func (h *ProjectHandler) List(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.UserIDFromContext(r.Context())
	if !ok {
		httputil.Error(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	projects, err := h.projects.ListByUser(r.Context(), userID)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}
	if projects == nil {
		projects = []domain.Project{}
	}

	httputil.JSON(w, http.StatusOK, projects)
}

// Get godoc
// @Summary Get project by ID
// @Tags projects
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Success 200 {object} domain.Project
// @Failure 404 {object} httputil.ErrorResponse
// @Router /projects/{projectID} [get]
func (h *ProjectHandler) Get(w http.ResponseWriter, r *http.Request) {
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

	httputil.JSON(w, http.StatusOK, project)
}

// Update godoc
// @Summary Update project
// @Tags projects
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param body body updateProjectRequest true "Project data"
// @Success 200 {object} domain.Project
// @Failure 404 {object} httputil.ErrorResponse
// @Router /projects/{projectID} [put]
func (h *ProjectHandler) Update(w http.ResponseWriter, r *http.Request) {
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

	existing, err := h.projects.GetByID(r.Context(), projectID)
	if err != nil {
		if errors.Is(err, repo.ErrProjectNotFound) {
			httputil.Error(w, http.StatusNotFound, "project not found")
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}
	if existing.UserID != userID {
		httputil.Error(w, http.StatusForbidden, "forbidden")
		return
	}

	var req updateProjectRequest
	if err := httputil.DecodeJSON(r, &req); err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if err := validate.Struct(req); err != nil {
		httputil.ErrorWithDetails(w, http.StatusUnprocessableEntity, "validation failed", err.Error())
		return
	}

	project, err := h.projects.Update(r.Context(), projectID, req.Name, req.Description)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	httputil.JSON(w, http.StatusOK, project)
}

// Delete godoc
// @Summary Delete project
// @Tags projects
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Success 204
// @Failure 404 {object} httputil.ErrorResponse
// @Router /projects/{projectID} [delete]
func (h *ProjectHandler) Delete(w http.ResponseWriter, r *http.Request) {
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

	existing, err := h.projects.GetByID(r.Context(), projectID)
	if err != nil {
		if errors.Is(err, repo.ErrProjectNotFound) {
			httputil.Error(w, http.StatusNotFound, "project not found")
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}
	if existing.UserID != userID {
		httputil.Error(w, http.StatusForbidden, "forbidden")
		return
	}

	if err := h.projects.Delete(r.Context(), projectID); err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	w.WriteHeader(http.StatusNoContent)
}
