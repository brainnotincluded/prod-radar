package detector

import (
	"math"
	"testing"
)

func TestComputeStats_Empty(t *testing.T) {
	s := ComputeStats(nil)
	if s.Mean != 0 || s.StdDev != 0 || s.Count != 0 {
		t.Errorf("empty input should return zero stats, got %+v", s)
	}
}

func TestComputeStats_Single(t *testing.T) {
	s := ComputeStats([]float64{5.0})
	if s.Mean != 5.0 {
		t.Errorf("mean should be 5, got %f", s.Mean)
	}
	if s.StdDev != 0 {
		t.Errorf("stddev of single value should be 0, got %f", s.StdDev)
	}
	if s.Count != 1 {
		t.Errorf("count should be 1, got %d", s.Count)
	}
}

func TestComputeStats_Multiple(t *testing.T) {
	s := ComputeStats([]float64{2, 4, 4, 4, 5, 5, 7, 9})
	if math.Abs(s.Mean-5.0) > 0.001 {
		t.Errorf("mean should be 5.0, got %f", s.Mean)
	}
	if math.Abs(s.StdDev-2.0) > 0.001 {
		t.Errorf("stddev should be 2.0, got %f", s.StdDev)
	}
	if s.Count != 8 {
		t.Errorf("count should be 8, got %d", s.Count)
	}
}

func TestIsSpike_NotEnoughData(t *testing.T) {
	s := Stats{Mean: 5, StdDev: 1, Count: 1}
	if IsSpike(100, s, 2.0) {
		t.Error("should not detect spike with count < 2")
	}
}

func TestIsSpike_ZeroStdDev(t *testing.T) {
	s := Stats{Mean: 5, StdDev: 0, Count: 5}
	if !IsSpike(6, s, 2.0) {
		t.Error("should detect spike when current > mean and stddev is 0")
	}
	if IsSpike(5, s, 2.0) {
		t.Error("should not detect spike when current == mean and stddev is 0")
	}
	if IsSpike(4, s, 2.0) {
		t.Error("should not detect spike when current < mean and stddev is 0")
	}
}

func TestIsSpike_NormalSpike(t *testing.T) {
	s := Stats{Mean: 10, StdDev: 2, Count: 10}
	if !IsSpike(14, s, 2.0) {
		t.Error("z-score=2.0 should trigger spike at threshold=2.0")
	}
	if !IsSpike(15, s, 2.0) {
		t.Error("z-score=2.5 should trigger spike at threshold=2.0")
	}
}

func TestIsSpike_NoSpike(t *testing.T) {
	s := Stats{Mean: 10, StdDev: 2, Count: 10}
	if IsSpike(11, s, 2.0) {
		t.Error("z-score=0.5 should not trigger spike at threshold=2.0")
	}
	if IsSpike(13, s, 2.0) {
		t.Error("z-score=1.5 should not trigger spike at threshold=2.0")
	}
}

func TestIsSpike_ExactThreshold(t *testing.T) {
	s := Stats{Mean: 10, StdDev: 5, Count: 10}
	if !IsSpike(20, s, 2.0) {
		t.Error("z-score=2.0 should trigger spike at threshold=2.0 (exact match)")
	}
}
