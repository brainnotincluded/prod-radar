package consumer

import (
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestFormatAlertMessage(t *testing.T) {
	spike := SpikeAlert{
		ProjectID:  uuid.New(),
		Current:    15.0,
		Mean:       5.0,
		StdDev:     2.5,
		Threshold:  2.0,
		DetectedAt: time.Now(),
	}

	msg := formatAlertMessage(spike)
	if !strings.Contains(msg, "Spike detected") {
		t.Error("message should contain 'Spike detected'")
	}
	if !strings.Contains(msg, "current=15.0") {
		t.Errorf("message should contain current=15.0, got: %s", msg)
	}
	if !strings.Contains(msg, "mean=5.0") {
		t.Errorf("message should contain mean=5.0, got: %s", msg)
	}
	if !strings.Contains(msg, "stddev=2.5") {
		t.Errorf("message should contain stddev=2.5, got: %s", msg)
	}
}

func TestFormatAlertMessage_ZeroValues(t *testing.T) {
	spike := SpikeAlert{}
	msg := formatAlertMessage(spike)
	if !strings.Contains(msg, "current=0.0") {
		t.Errorf("zero current should format as 0.0, got: %s", msg)
	}
}

func TestSpikeAlert_Fields(t *testing.T) {
	pID := uuid.New()
	spike := SpikeAlert{
		ProjectID:  pID,
		Current:    10,
		Mean:       5,
		StdDev:     2,
		Threshold:  2.0,
		DetectedAt: time.Now(),
	}
	if spike.ProjectID != pID {
		t.Error("project ID mismatch")
	}
	if spike.Current != 10 {
		t.Error("current mismatch")
	}
}
