package handler

import (
	"net/http"

	"github.com/brandradar/pkg/domain"
	"github.com/brandradar/pkg/httputil"
	"github.com/brandradar/services/api/repo"
)

type BrandCatalogHandler struct {
	catalog *repo.BrandCatalogRepo
}

func NewBrandCatalogHandler(catalog *repo.BrandCatalogRepo) *BrandCatalogHandler {
	return &BrandCatalogHandler{catalog: catalog}
}

// Search godoc
// @Summary Search brand catalog
// @Tags brand-catalog
// @Produce json
// @Param q query string true "Search query"
// @Param limit query int false "Limit" default(20)
// @Success 200 {array} domain.BrandCatalog
// @Router /brand-catalog/search [get]
func (h *BrandCatalogHandler) Search(w http.ResponseWriter, r *http.Request) {
	query := httputil.QueryString(r, "q", "")
	if query == "" {
		httputil.Error(w, http.StatusBadRequest, "query parameter 'q' is required")
		return
	}
	limit := httputil.QueryInt(r, "limit", 20)
	if limit > 100 {
		limit = 100
	}

	brands, err := h.catalog.Search(r.Context(), query, limit)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}
	if brands == nil {
		brands = []domain.BrandCatalog{}
	}

	httputil.JSON(w, http.StatusOK, brands)
}

// List godoc
// @Summary List brand catalog
// @Tags brand-catalog
// @Produce json
// @Param offset query int false "Offset" default(0)
// @Param limit query int false "Limit" default(50)
// @Success 200 {object} map[string]interface{}
// @Router /brand-catalog [get]
func (h *BrandCatalogHandler) List(w http.ResponseWriter, r *http.Request) {
	offset := httputil.QueryInt(r, "offset", 0)
	limit := httputil.QueryInt(r, "limit", 50)
	if limit > 100 {
		limit = 100
	}

	brands, total, err := h.catalog.List(r.Context(), offset, limit)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}
	if brands == nil {
		brands = []domain.BrandCatalog{}
	}

	httputil.JSON(w, http.StatusOK, map[string]interface{}{
		"data":  brands,
		"total": total,
	})
}
