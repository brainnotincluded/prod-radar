FROM golang:1.24-alpine AS builder

WORKDIR /build

COPY pkg/ ./pkg/
COPY services/enricher/ ./services/enricher/

RUN cd services/enricher && go mod download && \
    CGO_ENABLED=0 go build -o /build/enricher ./cmd

FROM alpine:3.19

RUN apk add --no-cache ca-certificates

COPY --from=builder /build/enricher /usr/local/bin/enricher

ENTRYPOINT ["enricher"]
