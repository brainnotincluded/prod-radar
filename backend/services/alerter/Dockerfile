FROM golang:1.24-alpine AS builder

WORKDIR /build

COPY pkg/ ./pkg/
COPY services/alerter/ ./services/alerter/

RUN cd services/alerter && go mod download && \
    CGO_ENABLED=0 go build -o /build/alerter ./cmd

FROM alpine:3.19

RUN apk add --no-cache ca-certificates

COPY --from=builder /build/alerter /usr/local/bin/alerter

ENTRYPOINT ["alerter"]
