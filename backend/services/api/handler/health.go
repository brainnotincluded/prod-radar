package handler

import (
	"context"
	"net/http"
	"time"

	"github.com/brandradar/pkg/httputil"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/nats-io/nats.go"
	"github.com/redis/go-redis/v9"
)

type HealthHandler struct {
	pool *pgxpool.Pool
	rdb  *redis.Client
	nc   *nats.Conn
}

func NewHealthHandler(pool *pgxpool.Pool, rdb *redis.Client, nc *nats.Conn) *HealthHandler {
	return &HealthHandler{pool: pool, rdb: rdb, nc: nc}
}

type HealthResponse struct {
	Status   string `json:"status"`
	Database string `json:"database"`
	Redis    string `json:"redis"`
	NATS     string `json:"nats"`
}

// Check godoc
// @Summary Health check
// @Tags health
// @Produce json
// @Success 200 {object} HealthResponse
// @Router /health [get]
func (h *HealthHandler) Check(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()

	dbStatus := "ok"
	if err := h.pool.Ping(ctx); err != nil {
		dbStatus = "unavailable"
	}

	redisStatus := "ok"
	if h.rdb != nil {
		if err := h.rdb.Ping(ctx).Err(); err != nil {
			redisStatus = "unavailable"
		}
	} else {
		redisStatus = "not_configured"
	}

	natsStatus := "ok"
	if h.nc != nil {
		if !h.nc.IsConnected() {
			natsStatus = "unavailable"
		}
	} else {
		natsStatus = "not_configured"
	}

	status := "ok"
	if dbStatus != "ok" || redisStatus == "unavailable" || natsStatus == "unavailable" {
		status = "degraded"
	}

	httputil.JSON(w, http.StatusOK, HealthResponse{
		Status:   status,
		Database: dbStatus,
		Redis:    redisStatus,
		NATS:     natsStatus,
	})
}
