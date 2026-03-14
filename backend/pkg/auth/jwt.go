package auth

import (
	"errors"
	"fmt"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

var (
	ErrInvalidToken = errors.New("auth: invalid token")
	ErrExpiredToken = errors.New("auth: token expired")
)

type TokenType string

const (
	TokenTypeAccess  TokenType = "access"
	TokenTypeRefresh TokenType = "refresh"
)

type Claims struct {
	UserID    uuid.UUID `json:"user_id"`
	TokenType TokenType `json:"token_type"`
	jwt.RegisteredClaims
}

type TokenPair struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
}

type JWTManager struct {
	secret          []byte
	accessDuration  time.Duration
	refreshDuration time.Duration
}

func NewJWTManager(secret string, accessDuration time.Duration) *JWTManager {
	return &JWTManager{
		secret:          []byte(secret),
		accessDuration:  accessDuration,
		refreshDuration: 7 * 24 * time.Hour,
	}
}

func (m *JWTManager) Generate(userID uuid.UUID) (string, error) {
	return m.generateToken(userID, TokenTypeAccess, m.accessDuration)
}

func (m *JWTManager) GenerateTokenPair(userID uuid.UUID) (*TokenPair, error) {
	access, err := m.generateToken(userID, TokenTypeAccess, m.accessDuration)
	if err != nil {
		return nil, err
	}

	refresh, err := m.generateToken(userID, TokenTypeRefresh, m.refreshDuration)
	if err != nil {
		return nil, err
	}

	return &TokenPair{AccessToken: access, RefreshToken: refresh}, nil
}

func (m *JWTManager) Refresh(refreshToken string) (*TokenPair, error) {
	claims, err := m.Verify(refreshToken)
	if err != nil {
		return nil, err
	}

	if claims.TokenType != TokenTypeRefresh {
		return nil, ErrInvalidToken
	}

	return m.GenerateTokenPair(claims.UserID)
}

func (m *JWTManager) generateToken(userID uuid.UUID, tokenType TokenType, duration time.Duration) (string, error) {
	now := time.Now()
	claims := Claims{
		UserID:    userID,
		TokenType: tokenType,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(now.Add(duration)),
			IssuedAt:  jwt.NewNumericDate(now),
			Subject:   userID.String(),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	signed, err := token.SignedString(m.secret)
	if err != nil {
		return "", fmt.Errorf("auth: sign token: %w", err)
	}

	return signed, nil
}

func (m *JWTManager) Verify(tokenStr string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenStr, &Claims{}, func(t *jwt.Token) (interface{}, error) {
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", t.Header["alg"])
		}
		return m.secret, nil
	})
	if err != nil {
		if errors.Is(err, jwt.ErrTokenExpired) {
			return nil, ErrExpiredToken
		}
		return nil, ErrInvalidToken
	}

	claims, ok := token.Claims.(*Claims)
	if !ok || !token.Valid {
		return nil, ErrInvalidToken
	}

	return claims, nil
}
