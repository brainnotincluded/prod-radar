FROM golang:1.24-alpine AS builder

WORKDIR /build

COPY pkg/ ./pkg/
COPY services/api/ ./services/api/

RUN cd services/api && go mod download && \
    CGO_ENABLED=0 go build -o /build/api ./cmd

FROM alpine:3.19

RUN apk add --no-cache ca-certificates

COPY --from=builder /build/api /usr/local/bin/api

ENTRYPOINT ["api"]
