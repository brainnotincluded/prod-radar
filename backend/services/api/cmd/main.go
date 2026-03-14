package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/brandradar/pkg/auth"
	"github.com/brandradar/pkg/config"
	"github.com/brandradar/pkg/messaging"
	"github.com/brandradar/pkg/postgres"
	_ "github.com/brandradar/services/api/docs"
	"github.com/brandradar/services/api/handler"
	"github.com/brandradar/services/api/repo"
	"github.com/brandradar/services/api/router"
	"github.com/brandradar/services/api/service"
	"github.com/redis/go-redis/v9"
)

// @title BrandRadar API
// @version 1.0
// @description Brand reputation monitoring system API
// @host localhost:8080
// @BasePath /api/v1
// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization
func main() {
	cfg, err := config.LoadBase()
	if err != nil {
		slog.Error("load config", "error", err)
		os.Exit(1)
	}

	ctx := context.Background()

	pool, err := postgres.NewPool(ctx, cfg.DatabaseURL)
	if err != nil {
		slog.Error("connect to postgres", "error", err)
		os.Exit(1)
	}
	defer pool.Close()

	redisOpts, err := redis.ParseURL(cfg.RedisURL)
	if err != nil {
		slog.Warn("parse redis url, health check will skip redis", "error", err)
	}
	var rdb *redis.Client
	if redisOpts != nil {
		rdb = redis.NewClient(redisOpts)
		defer rdb.Close()
	}

	nc, js, err := messaging.Connect(cfg.NatsURL)
	if err != nil {
		slog.Warn("connect to nats, health check will skip nats", "error", err)
	}

	jwtSecret := os.Getenv("JWT_SECRET")
	if jwtSecret == "" {
		jwtSecret = "brandradar-dev-secret-change-me"
	}
	jwtDuration := 24 * time.Hour
	jwtMgr := auth.NewJWTManager(jwtSecret, jwtDuration)

	userRepo := repo.NewUserRepo(pool)
	projectRepo := repo.NewProjectRepo(pool)
	sourceRepo := repo.NewSourceRepo(pool)
	catalogRepo := repo.NewBrandCatalogRepo(pool)
	brandRepo := repo.NewBrandRepo(pool)
	mentionRepo := repo.NewMentionRepo(pool)
	alertRepo := repo.NewAlertRepo(pool)
	alertConfigRepo := repo.NewAlertConfigRepo(pool)
	eventRepo := repo.NewEventRepo(pool)

	authSvc := service.NewAuthService(userRepo, jwtMgr)

	authHandler := handler.NewAuthHandler(authSvc)
	projectHandler := handler.NewProjectHandler(projectRepo)
	sourceHandler := handler.NewSourceHandler(sourceRepo, projectRepo, js)
	catalogHandler := handler.NewBrandCatalogHandler(catalogRepo)
	brandHandler := handler.NewBrandHandler(brandRepo, projectRepo)
	feedHandler := handler.NewFeedHandler(mentionRepo)
	analyticsHandler := handler.NewAnalyticsHandler(pool, mentionRepo)
	eventHandler := handler.NewEventHandler(eventRepo)
	alertHandler := handler.NewAlertHandler(alertRepo)
	alertConfigHandler := handler.NewAlertConfigHandler(alertConfigRepo, projectRepo)
	healthHandler := handler.NewHealthHandler(pool, rdb, nc)

	r := router.New(router.Deps{
		JWTManager:         jwtMgr,
		AuthHandler:        authHandler,
		ProjectHandler:     projectHandler,
		SourceHandler:      sourceHandler,
		CatalogHandler:     catalogHandler,
		FeedHandler:        feedHandler,
		AnalyticsHandler:   analyticsHandler,
		EventHandler:       eventHandler,
		AlertHandler:       alertHandler,
		AlertConfigHandler: alertConfigHandler,
		HealthHandler:      healthHandler,
		BrandHandler:       brandHandler,
	})

	port := os.Getenv("API_PORT")
	if port == "" {
		port = "8080"
	}

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      r,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		slog.Info("api server starting", "port", port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	slog.Info("shutting down server")
	shutdownCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		slog.Error("server shutdown error", "error", err)
	}
	slog.Info("server stopped")
}
