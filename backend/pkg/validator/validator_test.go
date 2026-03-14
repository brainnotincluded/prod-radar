package validator

import (
	"testing"
)

type testStruct struct {
	Email string `validate:"required,email"`
	Name  string `validate:"required,min=2"`
}

func TestValidate_Positive(t *testing.T) {
	s := testStruct{Email: "user@example.com", Name: "Test"}
	if err := Validate(s); err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestValidate_Negative_MissingRequired(t *testing.T) {
	s := testStruct{Email: "", Name: ""}
	err := Validate(s)
	if err == nil {
		t.Error("expected validation error for empty fields")
	}
}

func TestValidate_Negative_InvalidEmail(t *testing.T) {
	s := testStruct{Email: "not-an-email", Name: "Test"}
	err := Validate(s)
	if err == nil {
		t.Error("expected validation error for invalid email")
	}
}

func TestValidate_Edge_MinLength(t *testing.T) {
	s := testStruct{Email: "a@b.c", Name: "AB"}
	if err := Validate(s); err != nil {
		t.Errorf("expected no error for min length name, got %v", err)
	}

	s2 := testStruct{Email: "a@b.c", Name: "A"}
	if err := Validate(s2); err == nil {
		t.Error("expected error for name below min length")
	}
}

func TestGet_ReturnsInstance(t *testing.T) {
	v := Get()
	if v == nil {
		t.Error("Get() should return non-nil validator")
	}
}
