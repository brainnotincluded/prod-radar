package config

import "github.com/caarlos0/env/v11"

type Base struct {
	DatabaseURL string `env:"DATABASE_URL" envDefault:"postgres://brandradar:brandradar@localhost:5432/brandradar?sslmode=disable"`
	RedisURL    string `env:"REDIS_URL" envDefault:"redis://localhost:6379/0"`
	NatsURL     string `env:"NATS_URL" envDefault:"nats://localhost:4222"`
	LogLevel    string `env:"LOG_LEVEL" envDefault:"info"`
}

func LoadBase() (Base, error) {
	var cfg Base
	if err := env.Parse(&cfg); err != nil {
		return Base{}, err
	}
	return cfg, nil
}
