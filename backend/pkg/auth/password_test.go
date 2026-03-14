package auth

import (
	"testing"
)

func TestHashPassword_AndCheck(t *testing.T) {
	password := "MyStr0ngP@ss!"
	hash, err := HashPassword(password)
	if err != nil {
		t.Fatalf("HashPassword: %v", err)
	}
	if hash == "" {
		t.Fatal("expected non-empty hash")
	}
	if hash == password {
		t.Error("hash should not equal the plain password")
	}
	if !CheckPassword(hash, password) {
		t.Error("CheckPassword should return true for correct password")
	}
}

func TestCheckPassword_WrongPassword(t *testing.T) {
	hash, err := HashPassword("correct-password")
	if err != nil {
		t.Fatalf("HashPassword: %v", err)
	}
	if CheckPassword(hash, "wrong-password") {
		t.Error("CheckPassword should return false for wrong password")
	}
}

func TestCheckPassword_EmptyPassword(t *testing.T) {
	hash, err := HashPassword("some-password")
	if err != nil {
		t.Fatalf("HashPassword: %v", err)
	}
	if CheckPassword(hash, "") {
		t.Error("CheckPassword should return false for empty password")
	}
}

func TestHashPassword_DifferentHashesForSameInput(t *testing.T) {
	h1, _ := HashPassword("same-password")
	h2, _ := HashPassword("same-password")
	if h1 == h2 {
		t.Error("bcrypt should produce different hashes for the same input due to salt")
	}
}
