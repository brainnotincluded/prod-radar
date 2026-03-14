FROM golang:1.24-alpine AS builder

WORKDIR /build

COPY pkg/ ./pkg/
COPY services/collector/ ./services/collector/

RUN cd services/collector && go mod download && \
    CGO_ENABLED=0 go build -o /build/collector ./cmd

FROM alpine:3.19

RUN apk add --no-cache ca-certificates

COPY --from=builder /build/collector /usr/local/bin/collector

ENTRYPOINT ["collector"]
