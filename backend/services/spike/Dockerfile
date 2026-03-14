FROM golang:1.24-alpine AS builder

WORKDIR /build

COPY pkg/ ./pkg/
COPY services/spike/ ./services/spike/

RUN cd services/spike && go mod download && \
    CGO_ENABLED=0 go build -o /build/spike ./cmd

FROM alpine:3.19

RUN apk add --no-cache ca-certificates

COPY --from=builder /build/spike /usr/local/bin/spike

ENTRYPOINT ["spike"]
