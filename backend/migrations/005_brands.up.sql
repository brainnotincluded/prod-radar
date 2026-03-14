CREATE TABLE brands (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL UNIQUE REFERENCES projects(id) ON DELETE CASCADE,
    catalog_id INT REFERENCES brand_catalog(id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    keywords TEXT[] NOT NULL DEFAULT '{}',
    exclusions TEXT[] NOT NULL DEFAULT '{}',
    risk_words TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
