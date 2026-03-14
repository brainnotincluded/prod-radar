package domain

type HealthStatus string

const (
	HealthStatusOK       HealthStatus = "ok"
	HealthStatusDegraded HealthStatus = "degraded"
	HealthStatusDown     HealthStatus = "down"
)

type ComponentHealth struct {
	Name   string       `json:"name"`
	Status HealthStatus `json:"status"`
	Latency string     `json:"latency,omitempty"`
	Error   string      `json:"error,omitempty"`
}

type Health struct {
	Status     HealthStatus      `json:"status"`
	Components []ComponentHealth `json:"components"`
}
