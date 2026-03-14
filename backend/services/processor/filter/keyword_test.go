package filter

import "testing"

func TestMatchesKeywords(t *testing.T) {
	tests := []struct {
		name      string
		text      string
		keywords  []string
		wantMatch bool
	}{
		{"match exact", "Apple released new product", []string{"Apple"}, true},
		{"match case insensitive", "apple released new product", []string{"Apple"}, true},
		{"no match", "Microsoft released new product", []string{"Apple"}, false},
		{"empty keywords", "anything", nil, false},
		{"empty text", "", []string{"Apple"}, false},
		{"multiple keywords one match", "Apple news today", []string{"Samsung", "Apple"}, true},
		{"partial match", "Applesauce is good", []string{"Apple"}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MatchesKeywords(tt.text, tt.keywords)
			if (len(got) > 0) != tt.wantMatch {
				t.Errorf("MatchesKeywords() matched=%v, want matched=%v", len(got) > 0, tt.wantMatch)
			}
		})
	}
}

func TestMatchesKeywords_ReturnsAll(t *testing.T) {
	got := MatchesKeywords("Apple and Samsung news", []string{"Apple", "Samsung", "Google"})
	if len(got) != 2 {
		t.Errorf("expected 2 matched keywords, got %d", len(got))
	}
}

func TestMatchesExclusions(t *testing.T) {
	tests := []struct {
		name       string
		text       string
		exclusions []string
		want       bool
	}{
		{"excluded", "Buy Apple stock now", []string{"stock"}, true},
		{"not excluded", "Apple released iPhone", []string{"stock"}, false},
		{"empty exclusions", "anything", nil, false},
		{"case insensitive", "Buy APPLE Stock", []string{"stock"}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MatchesExclusions(tt.text, tt.exclusions); got != tt.want {
				t.Errorf("MatchesExclusions() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestShouldKeep(t *testing.T) {
	tests := []struct {
		name       string
		title      string
		content    string
		keywords   []string
		exclusions []string
		want       bool
	}{
		{"keep: matches keyword no exclusion", "Apple news", "new iphone released", []string{"Apple"}, []string{"stock"}, true},
		{"drop: matches exclusion", "Apple stock", "buy apple stock now", []string{"Apple"}, []string{"stock"}, false},
		{"drop: no keyword match", "Samsung news", "galaxy released", []string{"Apple"}, nil, false},
		{"keep: keyword in content", "Breaking news", "Apple announced today", []string{"Apple"}, nil, true},
		{"drop: empty keywords", "Apple news", "something", nil, nil, false},
		{"keep: keyword in title only", "Apple update", "no brand mentioned", []string{"Apple"}, nil, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ShouldKeep(tt.title, tt.content, tt.keywords, tt.exclusions); got != tt.want {
				t.Errorf("ShouldKeep() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatchRiskWords(t *testing.T) {
	t.Run("match risk words", func(t *testing.T) {
		got := MatchRiskWords("утечка данных и скандал", []string{"утечка", "скандал", "взлом"})
		if len(got) != 2 {
			t.Errorf("expected 2 matches, got %d", len(got))
		}
	})
	t.Run("no risk words", func(t *testing.T) {
		got := MatchRiskWords("happy news today", []string{"утечка", "скандал"})
		if len(got) != 0 {
			t.Errorf("expected 0 matches, got %d", len(got))
		}
	})
	t.Run("empty risk words list", func(t *testing.T) {
		got := MatchRiskWords("утечка data", nil)
		if len(got) != 0 {
			t.Errorf("expected 0 matches, got %d", len(got))
		}
	})
}

func TestAnalyze(t *testing.T) {
	t.Run("keep with risk words", func(t *testing.T) {
		r := Analyze("Apple скандал", "new product leak", []string{"Apple"}, nil, []string{"скандал", "leak"})
		if !r.Keep {
			t.Error("expected Keep=true")
		}
		if len(r.MatchedKeywords) != 1 {
			t.Errorf("expected 1 keyword, got %d", len(r.MatchedKeywords))
		}
		if len(r.MatchedRiskWords) != 2 {
			t.Errorf("expected 2 risk words, got %d", len(r.MatchedRiskWords))
		}
	})
	t.Run("drop no keyword match", func(t *testing.T) {
		r := Analyze("Samsung news", "content", []string{"Apple"}, nil, []string{"скандал"})
		if r.Keep {
			t.Error("expected Keep=false")
		}
	})
	t.Run("drop exclusion match", func(t *testing.T) {
		r := Analyze("Apple stock", "buy stock", []string{"Apple"}, []string{"stock"}, nil)
		if r.Keep {
			t.Error("expected Keep=false when exclusion matches")
		}
	})
}
