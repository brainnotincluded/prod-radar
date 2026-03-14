package domain

import (
	"encoding/json"
	"testing"

	"github.com/google/uuid"
)

func TestUserPasswordHashHidden(t *testing.T) {
	u := User{
		ID:           uuid.New(),
		Email:        "test@example.com",
		PasswordHash: "secret_hash",
	}
	data, err := json.Marshal(u)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m["password_hash"]; ok {
		t.Error("password_hash should not appear in JSON output")
	}
}

func TestSourceTypeConstants(t *testing.T) {
	types := []SourceType{SourceTypeRSS, SourceTypeWeb, SourceTypeTelegram}
	expected := []string{"rss", "web", "telegram"}
	for i, st := range types {
		if string(st) != expected[i] {
			t.Errorf("expected %q, got %q", expected[i], st)
		}
	}
}

func TestMentionStatusConstants(t *testing.T) {
	statuses := []MentionStatus{MentionStatusPendingML, MentionStatusEnriched, MentionStatusReady, MentionStatusDismissed}
	expected := []string{"pending_ml", "enriched", "ready", "dismissed"}
	for i, s := range statuses {
		if string(s) != expected[i] {
			t.Errorf("expected %q, got %q", expected[i], s)
		}
	}
}

func TestAlertStatusConstants(t *testing.T) {
	statuses := []AlertStatus{AlertStatusPending, AlertStatusSent, AlertStatusFailed}
	expected := []string{"pending", "sent", "failed"}
	for i, s := range statuses {
		if string(s) != expected[i] {
			t.Errorf("expected %q, got %q", expected[i], s)
		}
	}
}

func TestHealthStatusConstants(t *testing.T) {
	statuses := []HealthStatus{HealthStatusOK, HealthStatusDegraded, HealthStatusDown}
	expected := []string{"ok", "degraded", "down"}
	for i, s := range statuses {
		if string(s) != expected[i] {
			t.Errorf("expected %q, got %q", expected[i], s)
		}
	}
}

func TestBrandCatalogJSON(t *testing.T) {
	bc := BrandCatalog{
		ID:        1,
		Slug:      "test-brand",
		Name:      "Test Brand",
		SourceURL: "https://brandcatalog.ru/mark/test-brand",
	}
	data, err := json.Marshal(bc)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if m["slug"] != "test-brand" {
		t.Errorf("expected slug 'test-brand', got %v", m["slug"])
	}
	if m["name"] != "Test Brand" {
		t.Errorf("expected name 'Test Brand', got %v", m["name"])
	}
}

func TestBrandOptionalCatalogID(t *testing.T) {
	b := Brand{
		ID:        uuid.New(),
		ProjectID: uuid.New(),
		Name:      "MyBrand",
		Keywords:  []string{"keyword1"},
	}
	data, err := json.Marshal(b)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m["catalog_id"]; ok {
		t.Error("catalog_id should be omitted when nil")
	}
}

func TestProjectOptionalDescription(t *testing.T) {
	p := Project{
		ID:     uuid.New(),
		UserID: uuid.New(),
		Name:   "TestProject",
	}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m["description"]; ok {
		t.Error("description should be omitted when nil")
	}
}
