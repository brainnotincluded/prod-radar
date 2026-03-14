package router

import (
	"github.com/brandradar/pkg/auth"
	"github.com/brandradar/services/api/handler"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	httpSwagger "github.com/swaggo/http-swagger"
)

type Deps struct {
	JWTManager         *auth.JWTManager
	AuthHandler        *handler.AuthHandler
	ProjectHandler     *handler.ProjectHandler
	SourceHandler      *handler.SourceHandler
	CatalogHandler     *handler.BrandCatalogHandler
	FeedHandler        *handler.FeedHandler
	AnalyticsHandler   *handler.AnalyticsHandler
	EventHandler       *handler.EventHandler
	AlertHandler       *handler.AlertHandler
	AlertConfigHandler *handler.AlertConfigHandler
	HealthHandler      *handler.HealthHandler
	BrandHandler       *handler.BrandHandler
}

func New(d Deps) *chi.Mux {
	r := chi.NewRouter()

	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)

	r.Get("/health", d.HealthHandler.Check)
	r.Get("/swagger/*", httpSwagger.WrapHandler)

	r.Route("/api/v1", func(r chi.Router) {
		r.Post("/auth/register", d.AuthHandler.Register)
		r.Post("/auth/login", d.AuthHandler.Login)
		r.Post("/auth/refresh", d.AuthHandler.Refresh)

		r.Get("/brand-catalog", d.CatalogHandler.List)
		r.Get("/brand-catalog/search", d.CatalogHandler.Search)

		r.Group(func(r chi.Router) {
			r.Use(auth.Middleware(d.JWTManager))

			r.Get("/auth/me", d.AuthHandler.Me)

			r.Route("/projects", func(r chi.Router) {
				r.Post("/", d.ProjectHandler.Create)
				r.Get("/", d.ProjectHandler.List)

				r.Route("/{projectID}", func(r chi.Router) {
					r.Get("/", d.ProjectHandler.Get)
					r.Put("/", d.ProjectHandler.Update)
					r.Delete("/", d.ProjectHandler.Delete)

					r.Route("/sources", func(r chi.Router) {
						r.Post("/", d.SourceHandler.Create)
						r.Get("/", d.SourceHandler.List)
						r.Route("/{sourceID}", func(r chi.Router) {
							r.Put("/", d.SourceHandler.Update)
							r.Delete("/", d.SourceHandler.Delete)
							r.Post("/toggle", d.SourceHandler.Toggle)
						})
					})

					r.Route("/brand", func(r chi.Router) {
						r.Post("/", d.BrandHandler.Create)
						r.Get("/", d.BrandHandler.Get)
						r.Put("/", d.BrandHandler.Update)
						r.Delete("/", d.BrandHandler.Delete)
					})

					r.Route("/feed", func(r chi.Router) {
						r.Get("/", d.FeedHandler.List)
						r.Route("/{mentionID}", func(r chi.Router) {
							r.Get("/", d.FeedHandler.Get)
							r.Get("/duplicates", d.FeedHandler.Duplicates)
							r.Post("/dismiss", d.FeedHandler.Dismiss)
						})
					})

					r.Route("/analytics", func(r chi.Router) {
						r.Get("/", d.AnalyticsHandler.Summary)
						r.Get("/timeline", d.AnalyticsHandler.Timeline)
						r.Get("/sentiment", d.AnalyticsHandler.Sentiment)
						r.Get("/sources", d.AnalyticsHandler.Sources)
					})

					r.Get("/events", d.EventHandler.List)
					r.Get("/alerts", d.AlertHandler.List)

					r.Route("/alert-config", func(r chi.Router) {
						r.Get("/", d.AlertConfigHandler.Get)
						r.Put("/", d.AlertConfigHandler.Upsert)
					})
				})
			})
		})
	})

	return r
}
