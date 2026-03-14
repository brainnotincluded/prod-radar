package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	baseURL   = "https://brandcatalog.ru/country/rossiya"
	batchSize = 500
)

var linkRe = regexp.MustCompile(`href="/mark/([^"]+)">([^<]+)</a>`)

type brand struct {
	Slug string
	Name string
}

func main() {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://brandradar:brandradar@localhost:5432/brandradar?sslmode=disable"
	}

	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		slog.Error("connect to db", "error", err)
		os.Exit(1)
	}
	defer pool.Close()

	letters := buildLetterPages()
	slog.Info("starting brand catalog seed", "pages", len(letters))

	var allBrands []brand
	client := &http.Client{Timeout: 30 * time.Second}

	for _, letter := range letters {
		url := baseURL + "/" + letter
		brands, err := fetchBrands(client, url)
		if err != nil {
			slog.Warn("fetch page failed", "letter", letter, "error", err)
			continue
		}
		slog.Info("fetched", "letter", letter, "count", len(brands))
		allBrands = append(allBrands, brands...)
		time.Sleep(500 * time.Millisecond)
	}

	slog.Info("total brands parsed", "count", len(allBrands))

	if err := upsertBrands(ctx, pool, allBrands); err != nil {
		slog.Error("upsert brands", "error", err)
		os.Exit(1)
	}

	slog.Info("seed completed", "total", len(allBrands))
}

func buildLetterPages() []string {
	var pages []string
	for c := 'A'; c <= 'Z'; c++ {
		pages = append(pages, string(c))
	}
	cyrillic := []rune("АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ")
	for _, c := range cyrillic {
		pages = append(pages, string(c))
	}
	return pages
}

func fetchBrands(client *http.Client, url string) ([]brand, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Cookie", "beget=begetok")
	req.Header.Set("User-Agent", "BrandRadar-Seed/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	matches := linkRe.FindAllStringSubmatch(string(body), -1)
	seen := make(map[string]struct{})
	var brands []brand

	for _, m := range matches {
		slug := strings.TrimSpace(m[1])
		name := strings.TrimSpace(m[2])
		if _, ok := seen[slug]; ok {
			continue
		}
		seen[slug] = struct{}{}
		brands = append(brands, brand{Slug: slug, Name: name})
	}

	return brands, nil
}

func upsertBrands(ctx context.Context, pool *pgxpool.Pool, brands []brand) error {
	for i := 0; i < len(brands); i += batchSize {
		end := i + batchSize
		if end > len(brands) {
			end = len(brands)
		}
		batch := brands[i:end]

		tx, err := pool.Begin(ctx)
		if err != nil {
			return fmt.Errorf("begin tx: %w", err)
		}

		for _, b := range batch {
			sourceURL := "https://brandcatalog.ru/mark/" + b.Slug
			_, err := tx.Exec(ctx,
				`INSERT INTO brand_catalog (slug, name, source_url)
				 VALUES ($1, $2, $3)
				 ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, source_url = EXCLUDED.source_url`,
				b.Slug, b.Name, sourceURL,
			)
			if err != nil {
				_ = tx.Rollback(ctx)
				return fmt.Errorf("upsert brand %q: %w", b.Slug, err)
			}
		}

		if err := tx.Commit(ctx); err != nil {
			return fmt.Errorf("commit tx: %w", err)
		}

		slog.Info("batch upserted", "from", i, "to", end)
	}
	return nil
}
