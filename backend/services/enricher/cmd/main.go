package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/brandradar/pkg/config"
	"github.com/brandradar/pkg/messaging"
	"github.com/brandradar/pkg/postgres"
	"github.com/brandradar/services/enricher/consumer"
)

func main() {
	cfg, err := config.LoadBase()
	if err != nil {
		slog.Error("load config", "error", err)
		os.Exit(1)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pool, err := postgres.NewPool(ctx, cfg.DatabaseURL)
	if err != nil {
		slog.Error("connect to postgres", "error", err)
		os.Exit(1)
	}
	defer pool.Close()

	_, js, err := messaging.Connect(cfg.NatsURL)
	if err != nil {
		slog.Error("connect to nats", "error", err)
		os.Exit(1)
	}

	mlURL := os.Getenv("ML_SERVICE_URL")

	go func() {
		quit := make(chan os.Signal, 1)
		signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
		<-quit
		slog.Info("shutting down enricher")
		cancel()
	}()

	c := consumer.New(pool, js, mlURL)
	if err := c.Run(ctx); err != nil {
		slog.Error("enricher run", "error", err)
		os.Exit(1)
	}
}
