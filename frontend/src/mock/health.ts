import type { HealthStatus } from '../types';

export const mockHealthStatuses: HealthStatus[] = [
  { id: 'news', name: 'Новостные источники', type: 'collector', status: 'healthy', lastFetchAt: new Date('2024-03-12T19:45:00'), latency: 120 },
  { id: 'vk', name: 'ВКонтакте', type: 'collector', status: 'healthy', lastFetchAt: new Date('2024-03-12T19:40:00'), latency: 250 },
  { id: 'telegram', name: 'Telegram каналы', type: 'collector', status: 'degraded', lastFetchAt: new Date('2024-03-12T19:30:00'), latency: 890, message: 'Задержки из-за rate limit' },
  { id: 'forums', name: 'Форумы', type: 'collector', status: 'healthy', lastFetchAt: new Date('2024-03-12T19:42:00'), latency: 180 },
  { id: 'blogs', name: 'Блоги', type: 'collector', status: 'healthy', lastFetchAt: new Date('2024-03-12T19:38:00'), latency: 200 },
  { id: 'reviews', name: 'Отзывы', type: 'collector', status: 'down', lastFetchAt: new Date('2024-03-12T18:00:00'), message: 'Обновление API' },
  { id: 'sentiment', name: 'ML: Анализ тональности', type: 'ml', status: 'healthy', latency: 45 },
  { id: 'risk', name: 'ML: Оценка рисков', type: 'ml', status: 'healthy', latency: 52 },
  { id: 'clustering', name: 'ML: Кластеризация', type: 'ml', status: 'healthy', latency: 120 },
  { id: 'telegram_bot', name: 'Telegram бот', type: 'alert', status: 'healthy' },
  { id: 'email', name: 'Email рассылка', type: 'alert', status: 'healthy' },
  { id: 'webhook', name: 'Webhook', type: 'alert', status: 'degraded', message: 'Таймауты у 2 получателей' },
];
