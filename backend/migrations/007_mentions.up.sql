CREATE TABLE mentions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    external_id VARCHAR(500) NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    url VARCHAR(1000),
    published_at TIMESTAMPTZ NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    matched_keywords TEXT[] DEFAULT '{}',
    matched_risk_words TEXT[] DEFAULT '{}',
    sentiment_label VARCHAR(20),
    sentiment_score FLOAT,
    embedding vector(768),
    cluster_id UUID,
    is_duplicate BOOLEAN DEFAULT FALSE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending_ml'
        CHECK (status IN ('pending_ml', 'enriched', 'ready', 'dismissed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(source_id, external_id)
);

CREATE INDEX idx_mentions_project_status ON mentions(project_id, status);
CREATE INDEX idx_mentions_project_published ON mentions(project_id, published_at DESC);
CREATE INDEX idx_mentions_project_sentiment ON mentions(project_id, sentiment_label);
CREATE INDEX idx_mentions_cluster ON mentions(cluster_id);
CREATE INDEX idx_mentions_embedding ON mentions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
