package messaging

import (
	"testing"
	"time"
)

func TestBackoffDelay_FirstDelivery(t *testing.T) {
	d := backoffDelay(1)
	if d != 1*time.Second {
		t.Errorf("expected 1s, got %v", d)
	}
}

func TestBackoffDelay_MiddleDelivery(t *testing.T) {
	d := backoffDelay(3)
	if d != 15*time.Second {
		t.Errorf("expected 15s, got %v", d)
	}
}

func TestBackoffDelay_BeyondMax(t *testing.T) {
	d := backoffDelay(100)
	if d != 60*time.Second {
		t.Errorf("expected 60s (capped), got %v", d)
	}
}

func TestUnmarshal_ValidJSON(t *testing.T) {
	type sample struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	data := []byte(`{"name":"test","age":25}`)
	v, err := Unmarshal[sample](data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v.Name != "test" || v.Age != 25 {
		t.Errorf("unexpected value: %+v", v)
	}
}

func TestUnmarshal_InvalidJSON(t *testing.T) {
	type sample struct {
		Name string `json:"name"`
	}
	data := []byte(`{invalid}`)
	_, err := Unmarshal[sample](data)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestUnmarshal_EmptyJSON(t *testing.T) {
	type sample struct {
		Name string `json:"name"`
	}
	data := []byte(`{}`)
	v, err := Unmarshal[sample](data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v.Name != "" {
		t.Errorf("expected empty name, got %q", v.Name)
	}
}

func TestConsumerConfig_Defaults(t *testing.T) {
	c := NewConsumer(nil, ConsumerConfig{
		Stream:        "TEST",
		Durable:       "test-consumer",
		FilterSubject: "test.>",
	})
	if c.cfg.MaxDeliver != 5 {
		t.Errorf("expected MaxDeliver=5, got %d", c.cfg.MaxDeliver)
	}
	if c.cfg.AckWait != 30*time.Second {
		t.Errorf("expected AckWait=30s, got %v", c.cfg.AckWait)
	}
	if c.cfg.BatchSize != 10 {
		t.Errorf("expected BatchSize=10, got %d", c.cfg.BatchSize)
	}
}

func TestConsumerConfig_CustomValues(t *testing.T) {
	c := NewConsumer(nil, ConsumerConfig{
		Stream:        "TEST",
		Durable:       "test-consumer",
		FilterSubject: "test.>",
		MaxDeliver:    10,
		AckWait:       60 * time.Second,
		BatchSize:     50,
	})
	if c.cfg.MaxDeliver != 10 {
		t.Errorf("expected MaxDeliver=10, got %d", c.cfg.MaxDeliver)
	}
	if c.cfg.AckWait != 60*time.Second {
		t.Errorf("expected AckWait=60s, got %v", c.cfg.AckWait)
	}
	if c.cfg.BatchSize != 50 {
		t.Errorf("expected BatchSize=50, got %d", c.cfg.BatchSize)
	}
}
