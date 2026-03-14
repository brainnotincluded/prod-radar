package consumer

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/uuid"
)

// --- callMLService tests ---

func TestCallMLService_Success(t *testing.T) {
	expected := mlResponse{
		SentimentLabel: "positive",
		SentimentScore: 0.95,
		Embedding:      []float32{0.1, 0.2, 0.3},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/analyze" {
			t.Errorf("expected /analyze, got %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected application/json content type")
		}

		var req mlRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Text == "" {
			t.Error("expected non-empty text")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(expected)
	}))
	defer srv.Close()

	c := &Consumer{
		mlURL:    srv.URL,
		mlClient: srv.Client(),
	}

	resp, err := c.callMLService(context.Background(), "test text")
	if err != nil {
		t.Fatalf("callMLService: %v", err)
	}
	if resp.SentimentLabel != expected.SentimentLabel {
		t.Errorf("sentiment_label: got %s, want %s", resp.SentimentLabel, expected.SentimentLabel)
	}
	if resp.SentimentScore != expected.SentimentScore {
		t.Errorf("sentiment_score: got %f, want %f", resp.SentimentScore, expected.SentimentScore)
	}
	if len(resp.Embedding) != len(expected.Embedding) {
		t.Errorf("embedding length: got %d, want %d", len(resp.Embedding), len(expected.Embedding))
	}
}

func TestCallMLService_EmptyURL_ReturnsDefaults(t *testing.T) {
	c := &Consumer{
		mlURL:    "",
		mlClient: &http.Client{Timeout: 5 * time.Second},
	}

	resp, err := c.callMLService(context.Background(), "test text")
	if err != nil {
		t.Fatalf("callMLService: %v", err)
	}
	if resp.SentimentLabel != "neutral" {
		t.Errorf("expected neutral, got %s", resp.SentimentLabel)
	}
	if resp.SentimentScore != 0 {
		t.Errorf("expected 0, got %f", resp.SentimentScore)
	}
	if len(resp.Embedding) != 0 {
		t.Errorf("expected empty embedding, got %d", len(resp.Embedding))
	}
}

func TestCallMLService_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	c := &Consumer{
		mlURL:    srv.URL,
		mlClient: srv.Client(),
	}

	_, err := c.callMLService(context.Background(), "test")
	if err == nil {
		t.Fatal("expected error for 500 response")
	}
}

func TestCallMLService_InvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("not json"))
	}))
	defer srv.Close()

	c := &Consumer{
		mlURL:    srv.URL,
		mlClient: srv.Client(),
	}

	_, err := c.callMLService(context.Background(), "test")
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestCallMLService_Timeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
		json.NewEncoder(w).Encode(mlResponse{})
	}))
	defer srv.Close()

	c := &Consumer{
		mlURL:    srv.URL,
		mlClient: &http.Client{Timeout: 50 * time.Millisecond},
	}

	_, err := c.callMLService(context.Background(), "test")
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

// --- handle tests (unit, no DB) ---

func TestHandle_InvalidJSON(t *testing.T) {
	c := &Consumer{
		mlURL:    "",
		mlClient: &http.Client{},
	}

	err := c.handle(context.Background(), []byte("not json"))
	if err != nil {
		t.Errorf("invalid JSON should not return error (ack and skip), got %v", err)
	}
}

// --- FilteredMention serialization ---

func TestFilteredMention_JSON_RoundTrip(t *testing.T) {
	fm := FilteredMention{
		MentionID:        uuid.New(),
		SourceID:         uuid.New(),
		BrandID:          uuid.New(),
		ProjectID:        uuid.New(),
		ExternalID:       "ext-123",
		URL:              "https://example.com/article",
		Title:            "Test Title",
		Content:          "Test Content",
		PublishedAt:      time.Now().UTC().Truncate(time.Second),
		MatchedKeywords:  []string{"kw1", "kw2"},
		MatchedRiskWords: []string{"risk1"},
	}

	data, err := json.Marshal(fm)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded FilteredMention
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if decoded.MentionID != fm.MentionID {
		t.Errorf("mention_id mismatch")
	}
	if decoded.ExternalID != fm.ExternalID {
		t.Errorf("external_id mismatch")
	}
	if len(decoded.MatchedKeywords) != 2 {
		t.Errorf("expected 2 keywords, got %d", len(decoded.MatchedKeywords))
	}
	if len(decoded.MatchedRiskWords) != 1 {
		t.Errorf("expected 1 risk word, got %d", len(decoded.MatchedRiskWords))
	}
}

// --- mlResponse serialization ---

func TestMlResponse_JSON_RoundTrip(t *testing.T) {
	resp := mlResponse{
		SentimentLabel: "negative",
		SentimentScore: 0.85,
		Embedding:      make([]float32, 768),
	}

	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded mlResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if decoded.SentimentLabel != resp.SentimentLabel {
		t.Errorf("label mismatch")
	}
	if len(decoded.Embedding) != 768 {
		t.Errorf("embedding length: got %d, want 768", len(decoded.Embedding))
	}
}

func TestMlResponse_EmptyEmbedding(t *testing.T) {
	resp := mlResponse{
		SentimentLabel: "neutral",
		SentimentScore: 0.5,
	}

	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded mlResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if decoded.Embedding != nil {
		t.Errorf("expected nil embedding, got %v", decoded.Embedding)
	}
}
