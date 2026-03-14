package httputil

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
)

func DecodeJSON(r *http.Request, dst interface{}) error {
	if r.Body == nil {
		return fmt.Errorf("httputil: empty body")
	}
	defer r.Body.Close()

	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()

	if err := dec.Decode(dst); err != nil {
		return fmt.Errorf("httputil: decode json: %w", err)
	}
	return nil
}

func URLParamUUID(r *http.Request, key string) (uuid.UUID, error) {
	raw := chi.URLParam(r, key)
	if raw == "" {
		return uuid.UUID{}, fmt.Errorf("httputil: missing url param %q", key)
	}
	id, err := uuid.Parse(raw)
	if err != nil {
		return uuid.UUID{}, fmt.Errorf("httputil: invalid uuid %q: %w", key, err)
	}
	return id, nil
}

func QueryInt(r *http.Request, key string, defaultVal int) int {
	raw := r.URL.Query().Get(key)
	if raw == "" {
		return defaultVal
	}
	val, err := strconv.Atoi(raw)
	if err != nil {
		return defaultVal
	}
	return val
}

func QueryString(r *http.Request, key, defaultVal string) string {
	val := r.URL.Query().Get(key)
	if val == "" {
		return defaultVal
	}
	return val
}
