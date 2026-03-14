package handler

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRegisterRequest_Validation(t *testing.T) {
	tests := []struct {
		name    string
		body    registerRequest
		wantErr bool
	}{
		{"valid", registerRequest{Email: "a@b.com", Password: "123456"}, false},
		{"missing email", registerRequest{Password: "123456"}, true},
		{"invalid email", registerRequest{Email: "notanemail", Password: "123456"}, true},
		{"short password", registerRequest{Email: "a@b.com", Password: "123"}, true},
		{"missing password", registerRequest{Email: "a@b.com"}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validate.Struct(tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("validate.Struct() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestLoginRequest_Validation(t *testing.T) {
	tests := []struct {
		name    string
		body    loginRequest
		wantErr bool
	}{
		{"valid", loginRequest{Email: "a@b.com", Password: "123456"}, false},
		{"missing email", loginRequest{Password: "123456"}, true},
		{"invalid email", loginRequest{Email: "notanemail", Password: "123456"}, true},
		{"missing password", loginRequest{Email: "a@b.com"}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validate.Struct(tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("validate.Struct() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestAuthHandler_Register_BadJSON(t *testing.T) {
	h := &AuthHandler{}
	body := bytes.NewBufferString(`{invalid json}`)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/auth/register", body)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.Register(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rec.Code)
	}
}

func TestAuthHandler_Login_BadJSON(t *testing.T) {
	h := &AuthHandler{}
	body := bytes.NewBufferString(`{invalid json}`)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/auth/login", body)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.Login(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rec.Code)
	}
}

func TestAuthHandler_Register_ValidationError(t *testing.T) {
	h := &AuthHandler{}
	data, _ := json.Marshal(registerRequest{Email: "bad", Password: "1"})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/auth/register", bytes.NewReader(data))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.Register(rec, req)

	if rec.Code != http.StatusUnprocessableEntity {
		t.Errorf("expected 422, got %d", rec.Code)
	}
}

func TestAuthHandler_Me_NoContext(t *testing.T) {
	h := &AuthHandler{}
	req := httptest.NewRequest(http.MethodGet, "/api/v1/auth/me", nil)
	rec := httptest.NewRecorder()
	h.Me(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rec.Code)
	}
}
