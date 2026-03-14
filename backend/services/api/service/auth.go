package service

import (
	"context"
	"errors"
	"fmt"

	"github.com/brandradar/pkg/auth"
	"github.com/brandradar/pkg/domain"
	"github.com/brandradar/services/api/repo"
	"github.com/google/uuid"
)

var (
	ErrEmailTaken        = errors.New("service: email already taken")
	ErrInvalidCredentials = errors.New("service: invalid credentials")
)

type AuthService struct {
	users  *repo.UserRepo
	jwtMgr *auth.JWTManager
}

func NewAuthService(users *repo.UserRepo, jwtMgr *auth.JWTManager) *AuthService {
	return &AuthService{users: users, jwtMgr: jwtMgr}
}

type AuthTokens struct {
	AccessToken  string       `json:"access_token"`
	RefreshToken string       `json:"refresh_token"`
	User         *domain.User `json:"user"`
}

func (s *AuthService) Register(ctx context.Context, email, password string) (*AuthTokens, error) {
	_, err := s.users.GetByEmail(ctx, email)
	if err == nil {
		return nil, ErrEmailTaken
	}
	if !errors.Is(err, repo.ErrUserNotFound) {
		return nil, fmt.Errorf("service: check email: %w", err)
	}

	hash, err := auth.HashPassword(password)
	if err != nil {
		return nil, fmt.Errorf("service: hash password: %w", err)
	}

	user, err := s.users.Create(ctx, email, hash)
	if err != nil {
		return nil, fmt.Errorf("service: create user: %w", err)
	}

	pair, err := s.jwtMgr.GenerateTokenPair(user.ID)
	if err != nil {
		return nil, fmt.Errorf("service: generate tokens: %w", err)
	}

	return &AuthTokens{AccessToken: pair.AccessToken, RefreshToken: pair.RefreshToken, User: user}, nil
}

func (s *AuthService) Login(ctx context.Context, email, password string) (*AuthTokens, error) {
	user, err := s.users.GetByEmail(ctx, email)
	if err != nil {
		if errors.Is(err, repo.ErrUserNotFound) {
			return nil, ErrInvalidCredentials
		}
		return nil, fmt.Errorf("service: get user: %w", err)
	}

	if !auth.CheckPassword(user.PasswordHash, password) {
		return nil, ErrInvalidCredentials
	}

	pair, err := s.jwtMgr.GenerateTokenPair(user.ID)
	if err != nil {
		return nil, fmt.Errorf("service: generate tokens: %w", err)
	}

	return &AuthTokens{AccessToken: pair.AccessToken, RefreshToken: pair.RefreshToken, User: user}, nil
}

func (s *AuthService) Refresh(ctx context.Context, refreshToken string) (*AuthTokens, error) {
	pair, err := s.jwtMgr.Refresh(refreshToken)
	if err != nil {
		return nil, ErrInvalidCredentials
	}

	claims, err := s.jwtMgr.Verify(pair.AccessToken)
	if err != nil {
		return nil, fmt.Errorf("service: verify new token: %w", err)
	}

	user, err := s.users.GetByID(ctx, claims.UserID)
	if err != nil {
		return nil, fmt.Errorf("service: get user: %w", err)
	}

	return &AuthTokens{AccessToken: pair.AccessToken, RefreshToken: pair.RefreshToken, User: user}, nil
}

func (s *AuthService) Me(ctx context.Context, userID uuid.UUID) (*domain.User, error) {
	user, err := s.users.GetByID(ctx, userID)
	if err != nil {
		return nil, fmt.Errorf("service: get user: %w", err)
	}
	return user, nil
}
