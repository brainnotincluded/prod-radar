package detector

import (
	"math"
)

type Stats struct {
	Mean   float64
	StdDev float64
	Count  int
}

func ComputeStats(values []float64) Stats {
	n := len(values)
	if n == 0 {
		return Stats{}
	}

	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(n)

	var variance float64
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(n)

	return Stats{
		Mean:   mean,
		StdDev: math.Sqrt(variance),
		Count:  n,
	}
}

func IsSpike(current float64, stats Stats, threshold float64) bool {
	if stats.Count < 2 {
		return false
	}
	if stats.StdDev == 0 {
		return current > stats.Mean
	}
	zScore := (current - stats.Mean) / stats.StdDev
	return zScore >= threshold
}
