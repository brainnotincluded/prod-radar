package auth

import (
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestJWTManager_GenerateAndVerify(t *testing.T) {
	mgr := NewJWTManager("test-secret-key-32bytes!!", 1*time.Hour)
	userID := uuid.New()

	token, err := mgr.Generate(userID)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if token == "" {
		t.Fatal("expected non-empty token")
	}

	claims, err := mgr.Verify(token)
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if claims.UserID != userID {
		t.Errorf("expected user_id %s, got %s", userID, claims.UserID)
	}
}

func TestJWTManager_VerifyExpiredToken(t *testing.T) {
	mgr := NewJWTManager("test-secret", -1*time.Second)
	userID := uuid.New()

	token, err := mgr.Generate(userID)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	_, err = mgr.Verify(token)
	if err != ErrExpiredToken {
		t.Errorf("expected ErrExpiredToken, got %v", err)
	}
}

func TestJWTManager_VerifyInvalidToken(t *testing.T) {
	mgr := NewJWTManager("test-secret", 1*time.Hour)

	_, err := mgr.Verify("completely.invalid.token")
	if err != ErrInvalidToken {
		t.Errorf("expected ErrInvalidToken, got %v", err)
	}
}

func TestJWTManager_VerifyWrongSecret(t *testing.T) {
	mgr1 := NewJWTManager("secret-one", 1*time.Hour)
	mgr2 := NewJWTManager("secret-two", 1*time.Hour)

	token, err := mgr1.Generate(uuid.New())
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	_, err = mgr2.Verify(token)
	if err != ErrInvalidToken {
		t.Errorf("expected ErrInvalidToken for wrong secret, got %v", err)
	}
}

func TestJWTManager_VerifyEmptyString(t *testing.T) {
	mgr := NewJWTManager("test-secret", 1*time.Hour)

	_, err := mgr.Verify("")
	if err == nil {
		t.Error("expected error for empty token")
	}
}

func TestJWTManager_GenerateTokenPair(t *testing.T) {
	mgr := NewJWTManager("test-secret-pair", 1*time.Hour)
	userID := uuid.New()

	pair, err := mgr.GenerateTokenPair(userID)
	if err != nil {
		t.Fatalf("GenerateTokenPair: %v", err)
	}
	if pair.AccessToken == "" {
		t.Error("expected non-empty access token")
	}
	if pair.RefreshToken == "" {
		t.Error("expected non-empty refresh token")
	}
	if pair.AccessToken == pair.RefreshToken {
		t.Error("access and refresh tokens should be different")
	}

	accessClaims, err := mgr.Verify(pair.AccessToken)
	if err != nil {
		t.Fatalf("verify access: %v", err)
	}
	if accessClaims.TokenType != TokenTypeAccess {
		t.Errorf("expected access token type, got %s", accessClaims.TokenType)
	}
	if accessClaims.UserID != userID {
		t.Errorf("expected user %s, got %s", userID, accessClaims.UserID)
	}

	refreshClaims, err := mgr.Verify(pair.RefreshToken)
	if err != nil {
		t.Fatalf("verify refresh: %v", err)
	}
	if refreshClaims.TokenType != TokenTypeRefresh {
		t.Errorf("expected refresh token type, got %s", refreshClaims.TokenType)
	}
}

func TestJWTManager_Refresh_Success(t *testing.T) {
	mgr := NewJWTManager("test-secret-refresh", 1*time.Hour)
	userID := uuid.New()

	pair, err := mgr.GenerateTokenPair(userID)
	if err != nil {
		t.Fatalf("GenerateTokenPair: %v", err)
	}

	newPair, err := mgr.Refresh(pair.RefreshToken)
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if newPair.AccessToken == "" || newPair.RefreshToken == "" {
		t.Error("expected non-empty tokens from refresh")
	}

	claims, err := mgr.Verify(newPair.AccessToken)
	if err != nil {
		t.Fatalf("verify new access: %v", err)
	}
	if claims.UserID != userID {
		t.Errorf("expected user %s, got %s", userID, claims.UserID)
	}
}

func TestJWTManager_Refresh_WithAccessToken_Fails(t *testing.T) {
	mgr := NewJWTManager("test-secret-refresh-fail", 1*time.Hour)

	pair, err := mgr.GenerateTokenPair(uuid.New())
	if err != nil {
		t.Fatalf("GenerateTokenPair: %v", err)
	}

	_, err = mgr.Refresh(pair.AccessToken)
	if err != ErrInvalidToken {
		t.Errorf("expected ErrInvalidToken when refreshing with access token, got %v", err)
	}
}

func TestJWTManager_Refresh_InvalidToken(t *testing.T) {
	mgr := NewJWTManager("test-secret", 1*time.Hour)

	_, err := mgr.Refresh("garbage.token.value")
	if err == nil {
		t.Error("expected error for invalid refresh token")
	}
}
