package handler

import (
	"errors"
	"net/http"

	"github.com/brandradar/pkg/auth"
	"github.com/brandradar/pkg/httputil"
	"github.com/brandradar/services/api/service"
)

type AuthHandler struct {
	svc *service.AuthService
}

func NewAuthHandler(svc *service.AuthService) *AuthHandler {
	return &AuthHandler{svc: svc}
}

type registerRequest struct {
	Email    string `json:"email" validate:"required,email"`
	Password string `json:"password" validate:"required,min=6"`
}

type loginRequest struct {
	Email    string `json:"email" validate:"required,email"`
	Password string `json:"password" validate:"required"`
}

type refreshRequest struct {
	RefreshToken string `json:"refresh_token" validate:"required"`
}

// Register godoc
// @Summary Register a new user
// @Tags auth
// @Accept json
// @Produce json
// @Param body body registerRequest true "Registration data"
// @Success 201 {object} service.AuthTokens
// @Failure 409 {object} httputil.ErrorResponse
// @Failure 422 {object} httputil.ErrorResponse
// @Router /auth/register [post]
func (h *AuthHandler) Register(w http.ResponseWriter, r *http.Request) {
	var req registerRequest
	if err := httputil.DecodeJSON(r, &req); err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if err := validate.Struct(req); err != nil {
		httputil.ErrorWithDetails(w, http.StatusUnprocessableEntity, "validation failed", err.Error())
		return
	}

	tokens, err := h.svc.Register(r.Context(), req.Email, req.Password)
	if err != nil {
		if errors.Is(err, service.ErrEmailTaken) {
			httputil.Error(w, http.StatusConflict, "email already taken")
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	httputil.JSON(w, http.StatusCreated, tokens)
}

// Login godoc
// @Summary Login
// @Tags auth
// @Accept json
// @Produce json
// @Param body body loginRequest true "Login credentials"
// @Success 200 {object} service.AuthTokens
// @Failure 401 {object} httputil.ErrorResponse
// @Router /auth/login [post]
func (h *AuthHandler) Login(w http.ResponseWriter, r *http.Request) {
	var req loginRequest
	if err := httputil.DecodeJSON(r, &req); err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if err := validate.Struct(req); err != nil {
		httputil.ErrorWithDetails(w, http.StatusUnprocessableEntity, "validation failed", err.Error())
		return
	}

	tokens, err := h.svc.Login(r.Context(), req.Email, req.Password)
	if err != nil {
		if errors.Is(err, service.ErrInvalidCredentials) {
			httputil.Error(w, http.StatusUnauthorized, "invalid credentials")
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	httputil.JSON(w, http.StatusOK, tokens)
}

// Me godoc
// @Summary Get current user
// @Tags auth
// @Produce json
// @Security BearerAuth
// @Success 200 {object} domain.User
// @Failure 401 {object} httputil.ErrorResponse
// @Router /auth/me [get]
func (h *AuthHandler) Me(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.UserIDFromContext(r.Context())
	if !ok {
		httputil.Error(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	user, err := h.svc.Me(r.Context(), userID)
	if err != nil {
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	httputil.JSON(w, http.StatusOK, user)
}

// Refresh godoc
// @Summary Refresh access token
// @Tags auth
// @Accept json
// @Produce json
// @Param body body refreshRequest true "Refresh token"
// @Success 200 {object} service.AuthTokens
// @Failure 401 {object} httputil.ErrorResponse
// @Router /auth/refresh [post]
func (h *AuthHandler) Refresh(w http.ResponseWriter, r *http.Request) {
	var req refreshRequest
	if err := httputil.DecodeJSON(r, &req); err != nil {
		httputil.Error(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if err := validate.Struct(req); err != nil {
		httputil.ErrorWithDetails(w, http.StatusUnprocessableEntity, "validation failed", err.Error())
		return
	}

	tokens, err := h.svc.Refresh(r.Context(), req.RefreshToken)
	if err != nil {
		if errors.Is(err, service.ErrInvalidCredentials) {
			httputil.Error(w, http.StatusUnauthorized, "invalid refresh token")
			return
		}
		httputil.Error(w, http.StatusInternalServerError, "internal error")
		return
	}

	httputil.JSON(w, http.StatusOK, tokens)
}
