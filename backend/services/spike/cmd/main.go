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
	"github.com/brandradar/services/spike/consumer"
	"github.com/redis/go-redis/v9"
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

	redisOpts, err := redis.ParseURL(cfg.RedisURL)
	if err != nil {
		slog.Error("parse redis url", "error", err)
		os.Exit(1)
	}
	rdb := redis.NewClient(redisOpts)
	defer rdb.Close()

	go func() {
		quit := make(chan os.Signal, 1)
		signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
		<-quit
		slog.Info("shutting down spike detector")
		cancel()
	}()

	c := consumer.New(pool, js, rdb)
	if err := c.Run(ctx); err != nil {
		slog.Error("spike detector run", "error", err)
		os.Exit(1)
	}
}
