package auth

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestMiddleware_ValidToken(t *testing.T) {
	mgr := NewJWTManager("test-secret-32bytes!!", 1*time.Hour)
	userID := uuid.New()
	token, _ := mgr.Generate(userID)

	handler := Middleware(mgr)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id, ok := UserIDFromContext(r.Context())
		if !ok {
			t.Error("expected user_id in context")
			return
		}
		if id != userID {
			t.Errorf("expected %s, got %s", userID, id)
		}
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rec.Code)
	}
}

func TestMiddleware_MissingHeader(t *testing.T) {
	mgr := NewJWTManager("test-secret", 1*time.Hour)
	handler := Middleware(mgr)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("handler should not be called")
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rec.Code)
	}
}

func TestMiddleware_InvalidFormat(t *testing.T) {
	mgr := NewJWTManager("test-secret", 1*time.Hour)
	handler := Middleware(mgr)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("handler should not be called")
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Token something")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rec.Code)
	}
}

func TestMiddleware_ExpiredToken(t *testing.T) {
	mgr := NewJWTManager("test-secret", -1*time.Second)
	token, _ := mgr.Generate(uuid.New())

	handler := Middleware(mgr)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("handler should not be called")
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rec.Code)
	}
}

func TestUserIDFromContext_Missing(t *testing.T) {
	_, ok := UserIDFromContext(context.Background())
	if ok {
		t.Error("expected false for missing user_id")
	}
}
