import type { EventLog } from '../types';

export const mockEvents: EventLog[] = [
  { id: 'e1', timestamp: new Date('2024-03-12T19:45:00'), type: 'collector.fetch', description: 'Успешный сбор: 23 новых упоминания из новостных источников', severity: 'info' },
  { id: 'e2', timestamp: new Date('2024-03-12T19:40:00'), type: 'collector.fetch', description: 'Успешный сбор: 15 новых упоминаний из ВКонтакте', severity: 'info' },
  { id: 'e3', timestamp: new Date('2024-03-12T19:30:00'), type: 'alert.spike', description: 'Обнаружен всплеск негативных упоминаний: +45 за последний час', severity: 'warning' },
  { id: 'e4', timestamp: new Date('2024-03-12T19:15:00'), type: 'ml.complete', description: 'ML-анализ завершён: обработано 38 упоминаний за 2.3 сек', severity: 'info' },
  { id: 'e5', timestamp: new Date('2024-03-12T19:00:00'), type: 'collector.fetch', description: 'Успешный сбор: 31 новое упоминание из Telegram каналов', severity: 'info' },
  { id: 'e6', timestamp: new Date('2024-03-12T18:45:00'), type: 'alert.sent', description: 'Отправлено уведомление о критическом упоминании в Telegram', severity: 'info' },
  { id: 'e7', timestamp: new Date('2024-03-12T18:30:00'), type: 'collector.error', description: 'Ошибка подключения к API отзывов: таймаут соединения', severity: 'error' },
  { id: 'e8', timestamp: new Date('2024-03-12T18:15:00'), type: 'collector.fetch', description: 'Успешный сбор: 12 новых упоминаний с форумов', severity: 'info' },
  { id: 'e9', timestamp: new Date('2024-03-12T18:00:00'), type: 'alert.spike', description: 'Обнаружен всплеск рисковых упоминаний: кластер "сбой приложения" (8 упоминаний)', severity: 'warning' },
  { id: 'e10', timestamp: new Date('2024-03-12T17:45:00'), type: 'ml.complete', description: 'Кластеризация завершена: выявлено 3 новых кластера', severity: 'info' },
  { id: 'e11', timestamp: new Date('2024-03-12T17:30:00'), type: 'collector.fetch', description: 'Успешный сбор: 8 новых упоминаний из блогов', severity: 'info' },
  { id: 'e12', timestamp: new Date('2024-03-12T17:15:00'), type: 'alert.sent', description: 'Отправлен email alert: 2 адресатам', severity: 'info' },
  { id: 'e13', timestamp: new Date('2024-03-12T17:00:00'), type: 'collector.degraded', description: 'Сбор Telegram: увеличена задержка из-за rate limit API', severity: 'warning' },
  { id: 'e14', timestamp: new Date('2024-03-12T16:45:00'), type: 'collector.fetch', description: 'Успешный сбор: 19 новых упоминаний из новостных источников', severity: 'info' },
  { id: 'e15', timestamp: new Date('2024-03-12T16:30:00'), type: 'error.webhook', description: 'Webhook таймаут для endpoint https://company.ru/api/alerts', severity: 'error' },
  { id: 'e16', timestamp: new Date('2024-03-12T16:15:00'), type: 'ml.complete', description: 'Анализ тональности: обработано 42 упоминания', severity: 'info' },
  { id: 'e17', timestamp: new Date('2024-03-12T16:00:00'), type: 'collector.fetch', description: 'Успешный сбор: 27 новых упоминаний из социальных сетей', severity: 'info' },
  { id: 'e18', timestamp: new Date('2024-03-12T15:45:00'), type: 'alert.spike.critical', description: 'КРИТИЧЕСКИЙ: массовый сбой в приложении Т-Банка (12 связанных упоминаний)', severity: 'error' },
  { id: 'e19', timestamp: new Date('2024-03-12T15:30:00'), type: 'collector.down', description: 'Сборщик отзывов временно отключен: обновление API', severity: 'warning' },
  { id: 'e20', timestamp: new Date('2024-03-12T15:15:00'), type: 'alert.sent', description: 'Отправлено критическое уведомление в Telegram и email', severity: 'info' },
];
