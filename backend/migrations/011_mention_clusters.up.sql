CREATE TABLE mention_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    centroid vector(768) NOT NULL,
    mention_count INT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_mention_clusters_project ON mention_clusters(project_id);
CREATE INDEX idx_mention_clusters_centroid ON mention_clusters USING ivfflat (centroid vector_cosine_ops) WITH (lists = 50);
