package httputil

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestJSON_Success(t *testing.T) {
	rec := httptest.NewRecorder()
	data := map[string]string{"key": "value"}
	JSON(rec, http.StatusOK, data)

	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected application/json, got %q", ct)
	}

	var m map[string]string
	if err := json.NewDecoder(rec.Body).Decode(&m); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if m["key"] != "value" {
		t.Errorf("expected 'value', got %q", m["key"])
	}
}

func TestJSON_NilData(t *testing.T) {
	rec := httptest.NewRecorder()
	JSON(rec, http.StatusNoContent, nil)

	if rec.Code != http.StatusNoContent {
		t.Errorf("expected 204, got %d", rec.Code)
	}
	if rec.Body.Len() != 0 {
		t.Errorf("expected empty body, got %q", rec.Body.String())
	}
}

func TestError_Response(t *testing.T) {
	rec := httptest.NewRecorder()
	Error(rec, http.StatusBadRequest, "bad request")

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rec.Code)
	}

	var resp ErrorResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Error != "bad request" {
		t.Errorf("expected 'bad request', got %q", resp.Error)
	}
	if resp.Details != "" {
		t.Errorf("expected empty details, got %q", resp.Details)
	}
}

func TestErrorWithDetails_Response(t *testing.T) {
	rec := httptest.NewRecorder()
	ErrorWithDetails(rec, http.StatusUnprocessableEntity, "validation failed", "email is required")

	if rec.Code != http.StatusUnprocessableEntity {
		t.Errorf("expected 422, got %d", rec.Code)
	}

	var resp ErrorResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Error != "validation failed" {
		t.Errorf("expected 'validation failed', got %q", resp.Error)
	}
	if resp.Details != "email is required" {
		t.Errorf("expected 'email is required', got %q", resp.Details)
	}
}
