package messaging

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/nats-io/nats.go/jetstream"
)

type Publisher struct {
	js jetstream.JetStream
}

func NewPublisher(js jetstream.JetStream) *Publisher {
	return &Publisher{js: js}
}

func (p *Publisher) Publish(ctx context.Context, subject string, payload interface{}) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("publisher: marshal: %w", err)
	}

	_, err = p.js.Publish(ctx, subject, data)
	if err != nil {
		return fmt.Errorf("publisher: publish to %s: %w", subject, err)
	}

	return nil
}
