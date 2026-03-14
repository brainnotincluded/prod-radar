package messaging

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/nats-io/nats.go/jetstream"
)

type MessageHandler func(ctx context.Context, data []byte) error

type ConsumerConfig struct {
	Stream        string
	Subjects      []string
	Durable       string
	FilterSubject string
	MaxDeliver    int
	AckWait       time.Duration
	BatchSize     int
}

type Consumer struct {
	js  jetstream.JetStream
	cfg ConsumerConfig
}

func NewConsumer(js jetstream.JetStream, cfg ConsumerConfig) *Consumer {
	if cfg.MaxDeliver == 0 {
		cfg.MaxDeliver = 5
	}
	if cfg.AckWait == 0 {
		cfg.AckWait = 30 * time.Second
	}
	if cfg.BatchSize == 0 {
		cfg.BatchSize = 10
	}
	return &Consumer{js: js, cfg: cfg}
}

func (c *Consumer) Run(ctx context.Context, handler MessageHandler) error {
	stream, err := c.js.CreateOrUpdateStream(ctx, jetstream.StreamConfig{
		Name:     c.cfg.Stream,
		Subjects: c.cfg.Subjects,
	})
	if err != nil {
		return fmt.Errorf("consumer: create stream %s: %w", c.cfg.Stream, err)
	}

	cons, err := stream.CreateOrUpdateConsumer(ctx, jetstream.ConsumerConfig{
		Durable:       c.cfg.Durable,
		FilterSubject: c.cfg.FilterSubject,
		AckPolicy:     jetstream.AckExplicitPolicy,
		MaxDeliver:    c.cfg.MaxDeliver,
		AckWait:       c.cfg.AckWait,
	})
	if err != nil {
		return fmt.Errorf("consumer: create consumer %s: %w", c.cfg.Durable, err)
	}

	slog.Info("consumer started", "durable", c.cfg.Durable, "subject", c.cfg.FilterSubject)

	iter, err := cons.Messages(jetstream.PullMaxMessages(c.cfg.BatchSize))
	if err != nil {
		return fmt.Errorf("consumer: messages iterator: %w", err)
	}
	defer iter.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.Info("consumer stopped", "durable", c.cfg.Durable)
			return nil
		default:
		}

		msg, err := iter.Next()
		if err != nil {
			slog.Warn("consumer: next message", "durable", c.cfg.Durable, "error", err)
			continue
		}

		meta, _ := msg.Metadata()
		deliveryCount := 1
		if meta != nil {
			deliveryCount = int(meta.NumDelivered)
		}

		if err := handler(ctx, msg.Data()); err != nil {
			slog.Warn("consumer: handler error",
				"durable", c.cfg.Durable,
				"delivery", deliveryCount,
				"error", err,
			)

			if deliveryCount >= c.cfg.MaxDeliver {
				slog.Error("consumer: max deliveries reached, sending to DLQ",
					"durable", c.cfg.Durable,
					"delivery", deliveryCount,
				)
				_ = msg.Term()
			} else {
				_ = msg.NakWithDelay(backoffDelay(deliveryCount))
			}
			continue
		}

		_ = msg.Ack()
	}
}

func backoffDelay(delivery int) time.Duration {
	delays := []time.Duration{
		1 * time.Second,
		5 * time.Second,
		15 * time.Second,
		30 * time.Second,
		60 * time.Second,
	}
	if delivery-1 < len(delays) {
		return delays[delivery-1]
	}
	return delays[len(delays)-1]
}

func Unmarshal[T any](data []byte) (T, error) {
	var v T
	if err := json.Unmarshal(data, &v); err != nil {
		return v, fmt.Errorf("messaging: unmarshal: %w", err)
	}
	return v, nil
}
