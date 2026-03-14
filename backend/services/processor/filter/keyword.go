package filter

import (
	"strings"
)

type Result struct {
	Keep             bool
	MatchedKeywords  []string
	MatchedRiskWords []string
}

func MatchesKeywords(text string, keywords []string) []string {
	lower := strings.ToLower(text)
	var matched []string
	for _, kw := range keywords {
		if strings.Contains(lower, strings.ToLower(kw)) {
			matched = append(matched, kw)
		}
	}
	return matched
}

func MatchesExclusions(text string, exclusions []string) bool {
	lower := strings.ToLower(text)
	for _, ex := range exclusions {
		if strings.Contains(lower, strings.ToLower(ex)) {
			return true
		}
	}
	return false
}

func MatchRiskWords(text string, riskWords []string) []string {
	lower := strings.ToLower(text)
	var matched []string
	for _, rw := range riskWords {
		if strings.Contains(lower, strings.ToLower(rw)) {
			matched = append(matched, rw)
		}
	}
	return matched
}

func Analyze(title, content string, keywords, exclusions, riskWords []string) Result {
	combined := title + " " + content

	matchedKw := MatchesKeywords(combined, keywords)
	if len(matchedKw) == 0 {
		return Result{Keep: false}
	}
	if MatchesExclusions(combined, exclusions) {
		return Result{Keep: false}
	}

	matchedRisk := MatchRiskWords(combined, riskWords)

	return Result{
		Keep:             true,
		MatchedKeywords:  matchedKw,
		MatchedRiskWords: matchedRisk,
	}
}

func ShouldKeep(title, content string, keywords, exclusions []string) bool {
	combined := title + " " + content
	if len(MatchesKeywords(combined, keywords)) == 0 {
		return false
	}
	if MatchesExclusions(combined, exclusions) {
		return false
	}
	return true
}
