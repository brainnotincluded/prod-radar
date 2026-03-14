package handler

import (
	"errors"
	"net/http"

	"github.com/brandradar/pkg/auth"
	"github.com/brandradar/pkg/domain"
	"github.com/brandradar/pkg/httputil"
	"github.com/brandradar/services/api/repo"
)

type BrandHandler struct {
	brands   *repo.BrandRepo
	projects *repo.ProjectRepo
}

func NewBrandHandler(brands *repo.BrandRepo, projects *repo.ProjectRepo) *BrandHandler {
	return &BrandHandler{brands: brands, projects: projects}
}

type createBrandRequest struct {
	CatalogID  *int     `json:"catalog_id"`
	Name       string   `json:"name" validate:"required,min=1,max=255"`
	Keywords   []string `json:"keywords" validate:"required,min=1"`
	Exclusions []string `json:"exclusions"`
	RiskWords  []string `json:"risk_words"`
}

type updateBrandRequest struct {
	CatalogID  *int     `json:"catalog_id"`
	Name       string   `json:"name" validate:"required,min=1,max=255"`
	Keywords   []string `json:"keywords" validate:"required,min=1"`
	Exclusions []string `json:"exclusions"`
	RiskWords  []string `json:"risk_words"`
}

// Create godoc
// @Summary Create brand for project
// @Tags brands
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param body body createBrandRequest true "Brand data"
// @Success 201 {object} domain.Brand
// @Failure 422 {object} httputil.ErrorResponse
// @Router /projects/{projectID}/brand [post]
func (h *BrandHandler) Create(w http.ResponseWriter, r *http.Request) {
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

	var req createBrandRequest
	if err := httputil.DecodeJSON(r, &req); err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if err := validate.Struct(req); err != nil {
		httputil.ErrorWithDetails(w, http.StatusUnprocessableEntity, "validation failed", err.Error())
		return
	}

	if req.Exclusions == nil {
		req.Exclusions = []string{}
	}

	brand, err := h.brands.Create(r.Context(), projectID, req.CatalogID, req.Name, req.Keywords, req.Exclusions, req.RiskWords)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	httputil.JSON(w, http.StatusCreated, brand)
}

// Get godoc
// @Summary Get brand settings for project
// @Tags brands
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Success 200 {object} domain.Brand
// @Failure 404 {object} httputil.ErrorResponse
// @Router /projects/{projectID}/brand [get]
func (h *BrandHandler) Get(w http.ResponseWriter, r *http.Request) {
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

	brand, err := h.brands.GetByProjectID(r.Context(), projectID)
	if err != nil {
		if errors.Is(err, repo.ErrBrandNotFound) {
			httputil.JSON(w, http.StatusOK, []domain.Brand{})
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	httputil.JSON(w, http.StatusOK, brand)
}

// Update godoc
// @Summary Update brand settings
// @Tags brands
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Param body body updateBrandRequest true "Brand data"
// @Success 200 {object} domain.Brand
// @Failure 404 {object} httputil.ErrorResponse
// @Router /projects/{projectID}/brand [put]
func (h *BrandHandler) Update(w http.ResponseWriter, r *http.Request) {
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

	var req updateBrandRequest
	if err := httputil.DecodeJSON(r, &req); err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if err := validate.Struct(req); err != nil {
		httputil.ErrorWithDetails(w, http.StatusUnprocessableEntity, "validation failed", err.Error())
		return
	}

	if req.Exclusions == nil {
		req.Exclusions = []string{}
	}

	brand, err := h.brands.Update(r.Context(), projectID, req.CatalogID, req.Name, req.Keywords, req.Exclusions, req.RiskWords)
	if err != nil {
		if errors.Is(err, repo.ErrBrandNotFound) {
			httputil.Error(w, http.StatusNotFound, "brand not found")
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	httputil.JSON(w, http.StatusOK, brand)
}

// Delete godoc
// @Summary Delete brand
// @Tags brands
// @Security BearerAuth
// @Param projectID path string true "Project ID"
// @Success 204
// @Failure 404 {object} httputil.ErrorResponse
// @Router /projects/{projectID}/brand [delete]
func (h *BrandHandler) Delete(w http.ResponseWriter, r *http.Request) {
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

	if err := h.brands.Delete(r.Context(), projectID); err != nil {
		if errors.Is(err, repo.ErrBrandNotFound) {
			httputil.Error(w, http.StatusNotFound, "brand not found")
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	w.WriteHeader(http.StatusNoContent)
}
