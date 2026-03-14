import type { AnalyticsData } from '../types';

export const mockAnalytics: AnalyticsData = {
  sentimentOverTime: [
    { date: '2024-03-06', positive: 45, negative: 23, neutral: 32 },
    { date: '2024-03-07', positive: 52, negative: 18, neutral: 35 },
    { date: '2024-03-08', positive: 38, negative: 45, neutral: 28 },
    { date: '2024-03-09', positive: 41, negative: 32, neutral: 38 },
    { date: '2024-03-10', positive: 48, negative: 28, neutral: 31 },
    { date: '2024-03-11', positive: 35, negative: 52, neutral: 25 },
    { date: '2024-03-12', positive: 42, negative: 38, neutral: 35 },
  ],
  volumeByDay: [
    { date: '2024-03-06', count: 156 },
    { date: '2024-03-07', count: 142 },
    { date: '2024-03-08', count: 189 },
    { date: '2024-03-09', count: 134 },
    { date: '2024-03-10', count: 167 },
    { date: '2024-03-11', count: 245 },
    { date: '2024-03-12', count: 198 },
  ],
  sourceDistribution: [
    { source: 'social', count: 456 },
    { source: 'news', count: 234 },
    { source: 'forum', count: 189 },
    { source: 'blog', count: 87 },
    { source: 'review', count: 156 },
  ],
  topRiskTopics: [
    { topic: 'Сбои в приложении', count: 45, trend: 'up' },
    { topic: 'Блокировка счетов', count: 32, trend: 'up' },
    { topic: 'Комиссии', count: 28, trend: 'stable' },
    { topic: 'Техподдержка', count: 24, trend: 'down' },
    { topic: 'Переводы', count: 19, trend: 'up' },
  ],
  spikeHistory: [
    { date: '2024-03-08T10:30:00', severity: 'high', mentions: 89 },
    { date: '2024-03-11T14:20:00', severity: 'critical', mentions: 156 },
    { date: '2024-03-12T08:30:00', severity: 'medium', mentions: 67 },
  ],
  keyMetrics: {
    totalMentions: 1123,
    avgSentiment: -0.15,
    spikeCount: 3,
    avgResponseTime: 4.2,
  },
};
