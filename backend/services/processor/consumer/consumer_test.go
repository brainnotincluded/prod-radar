package consumer

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/brandradar/pkg/domain"
	"github.com/brandradar/services/processor/filter"
	"github.com/google/uuid"
)

// --- nullString ---

func TestNullString_NonEmpty(t *testing.T) {
	s := "hello"
	got := nullString(s)
	if got == nil {
		t.Fatal("expected non-nil pointer")
	}
	if *got != s {
		t.Errorf("expected %q, got %q", s, *got)
	}
}

func TestNullString_Empty(t *testing.T) {
	got := nullString("")
	if got != nil {
		t.Errorf("expected nil for empty string, got %q", *got)
	}
}

// --- RawMention serialization ---

func TestRawMention_JSON_RoundTrip(t *testing.T) {
	rm := RawMention{
		SourceID:    uuid.New(),
		BrandID:     uuid.New(),
		ProjectID:   uuid.New(),
		ExternalID:  "ext-456",
		URL:         "https://example.com",
		Title:       "Test",
		Content:     "Content here",
		Author:      "Author",
		PublishedAt: time.Now().UTC().Truncate(time.Second),
	}

	data, err := json.Marshal(rm)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded RawMention
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if decoded.SourceID != rm.SourceID {
		t.Errorf("source_id mismatch")
	}
	if decoded.ExternalID != rm.ExternalID {
		t.Errorf("external_id mismatch: got %q, want %q", decoded.ExternalID, rm.ExternalID)
	}
	if decoded.Author != rm.Author {
		t.Errorf("author mismatch")
	}
}

func TestRawMention_JSON_EmptyAuthor(t *testing.T) {
	rm := RawMention{
		SourceID:    uuid.New(),
		BrandID:     uuid.New(),
		ProjectID:   uuid.New(),
		ExternalID:  "ext-789",
		URL:         "https://example.com",
		Title:       "Test",
		Content:     "Content",
		PublishedAt: time.Now().UTC().Truncate(time.Second),
	}

	data, err := json.Marshal(rm)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var m map[string]interface{}
	json.Unmarshal(data, &m)
	if _, ok := m["author"]; ok {
		t.Error("author should be omitted when empty")
	}
}

// --- brandSettings serialization ---

func TestBrandSettings_JSON_RoundTrip(t *testing.T) {
	bs := brandSettings{
		Keywords:   []string{"keyword1", "keyword2"},
		Exclusions: []string{"excl1"},
		RiskWords:  []string{"risk1", "risk2"},
	}

	data, err := json.Marshal(bs)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded brandSettings
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(decoded.Keywords) != 2 {
		t.Errorf("keywords length: got %d, want 2", len(decoded.Keywords))
	}
	if len(decoded.Exclusions) != 1 {
		t.Errorf("exclusions length: got %d, want 1", len(decoded.Exclusions))
	}
	if len(decoded.RiskWords) != 2 {
		t.Errorf("risk_words length: got %d, want 2", len(decoded.RiskWords))
	}
}

func TestBrandSettings_EmptySlices(t *testing.T) {
	bs := brandSettings{}

	data, err := json.Marshal(bs)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded brandSettings
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if decoded.Keywords != nil {
		t.Errorf("expected nil keywords")
	}
}

// --- handle edge cases ---

func TestHandle_InvalidJSON(t *testing.T) {
	c := &Consumer{}

	err := c.handle(nil, []byte("not json"))
	if err != nil {
		t.Errorf("invalid JSON should not return error (skip), got %v", err)
	}
}

// --- DefaultRiskWords fallback ---

func TestDefaultRiskWords_NotEmpty(t *testing.T) {
	if len(domain.DefaultRiskWords) == 0 {
		t.Fatal("DefaultRiskWords should not be empty")
	}
}

// --- filter.Analyze integration ---

func TestFilterAnalyze_KeepWithRiskWords(t *testing.T) {
	r := filter.Analyze("Лукойл скандал", "Компания Лукойл попала в скандал",
		[]string{"Лукойл"}, []string{}, []string{"скандал", "штраф"})

	if !r.Keep {
		t.Fatal("expected Keep=true")
	}
	if len(r.MatchedKeywords) == 0 {
		t.Error("expected matched keywords")
	}
	if len(r.MatchedRiskWords) == 0 {
		t.Error("expected matched risk words")
	}
}

func TestFilterAnalyze_ExclusionFiltersOut(t *testing.T) {
	r := filter.Analyze("Лукойл реклама", "Компания Лукойл реклама",
		[]string{"Лукойл"}, []string{"реклама"}, []string{})

	if r.Keep {
		t.Error("expected Keep=false due to exclusion")
	}
}

func TestFilterAnalyze_NoKeywordMatch(t *testing.T) {
	r := filter.Analyze("Яндекс новости", "Яндекс выпустил обновление",
		[]string{"Лукойл"}, []string{}, []string{})

	if r.Keep {
		t.Error("expected Keep=false, no keyword match")
	}
}
