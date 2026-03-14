package config

import (
	"os"
	"testing"
)

func TestLoadBase_Defaults(t *testing.T) {
	cfg, err := LoadBase()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.DatabaseURL == "" {
		t.Error("DatabaseURL should have a default value")
	}
	if cfg.RedisURL == "" {
		t.Error("RedisURL should have a default value")
	}
	if cfg.NatsURL == "" {
		t.Error("NatsURL should have a default value")
	}
	if cfg.LogLevel != "info" {
		t.Errorf("LogLevel should default to 'info', got %q", cfg.LogLevel)
	}
}

func TestLoadBase_EnvOverride(t *testing.T) {
	os.Setenv("DATABASE_URL", "postgres://test:test@localhost/test")
	os.Setenv("LOG_LEVEL", "debug")
	defer func() {
		os.Unsetenv("DATABASE_URL")
		os.Unsetenv("LOG_LEVEL")
	}()

	cfg, err := LoadBase()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.DatabaseURL != "postgres://test:test@localhost/test" {
		t.Errorf("expected overridden DATABASE_URL, got %q", cfg.DatabaseURL)
	}
	if cfg.LogLevel != "debug" {
		t.Errorf("expected LOG_LEVEL=debug, got %q", cfg.LogLevel)
	}
}

func TestLoadBase_EmptyEnvUsesDefault(t *testing.T) {
	os.Unsetenv("DATABASE_URL")
	os.Unsetenv("REDIS_URL")
	os.Unsetenv("NATS_URL")
	os.Unsetenv("LOG_LEVEL")

	cfg, err := LoadBase()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.DatabaseURL != "postgres://brandradar:brandradar@localhost:5432/brandradar?sslmode=disable" {
		t.Errorf("expected default DATABASE_URL, got %q", cfg.DatabaseURL)
	}
}
