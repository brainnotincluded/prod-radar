export type MentionSource = 'news' | 'social' | 'blog' | 'forum' | 'review';
export type Sentiment = 'positive' | 'negative' | 'neutral';
export type RiskLevel = 'critical' | 'high' | 'medium' | 'low' | 'none';

export interface Project {
  id: string;
  name: string;
  keywords: string[];
  exclusions: string[];
  riskWords: string[];
  createdAt: Date;
}

export interface Mention {
  id: string;
  projectId: string;
  title: string;
  content: string;
  source: MentionSource;
  sourceUrl: string;
  sourceName: string;
  sentiment: Sentiment;
  riskLevel: RiskLevel;
  relevance: number;
  reach: number;
  publishedAt: Date;
  collectedAt: Date;
  author?: string;
  highlights: string[];
  mlConfidence: number;
  clusterId?: string;
  clusterSize?: number;
}

export interface AnalyticsData {
  sentimentOverTime: { date: string; positive: number; negative: number; neutral: number }[];
  volumeByDay: { date: string; count: number }[];
  sourceDistribution: { source: MentionSource; count: number }[];
  topRiskTopics: { topic: string; count: number; trend: 'up' | 'down' | 'stable' }[];
  spikeHistory: { date: string; severity: 'critical' | 'high' | 'medium'; mentions: number }[];
  keyMetrics: {
    totalMentions: number;
    avgSentiment: number;
    spikeCount: number;
    avgResponseTime: number;
  };
}

export interface HealthStatus {
  id: string;
  name: string;
  type: 'collector' | 'ml' | 'alert';
  status: 'healthy' | 'degraded' | 'down';
  lastFetchAt?: Date;
  latency?: number;
  message?: string;
}

export type EventSeverity = 'info' | 'warning' | 'error';

export interface EventLog {
  id: string;
  timestamp: Date;
  type: string;
  description: string;
  severity: EventSeverity;
  metadata?: Record<string, unknown>;
}

export interface AlertConfig {
  spikeThreshold: number;
  cooldownMinutes: number;
  channels: {
    telegram: boolean;
    email: boolean;
    webhook: boolean;
    inapp: boolean;
  };
  telegramChatId?: string;
  emailAddresses?: string[];
  webhookUrl?: string;
}

export interface ProjectConfig {
  keywords: string[];
  exclusions: string[];
  riskWords: string[];
}

export type TimeRange = 'today' | '3days' | 'week' | 'month';

export interface FilterState {
  sentiments: Sentiment[];
  sources: MentionSource[];
  riskLevels: RiskLevel[];
  minRelevance: number;
  timeRange: TimeRange;
}
