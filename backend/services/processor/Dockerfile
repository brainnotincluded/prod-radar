FROM golang:1.24-alpine AS builder

WORKDIR /build

COPY pkg/ ./pkg/
COPY services/processor/ ./services/processor/

RUN cd services/processor && go mod download && \
    CGO_ENABLED=0 go build -o /build/processor ./cmd

FROM alpine:3.19

RUN apk add --no-cache ca-certificates

COPY --from=builder /build/processor /usr/local/bin/processor

ENTRYPOINT ["processor"]
