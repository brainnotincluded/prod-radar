import type { AlertConfig } from '../types';

export const defaultAlertConfig: AlertConfig = {
  spikeThreshold: 25,
  cooldownMinutes: 30,
  channels: {
    telegram: true,
    email: true,
    webhook: false,
    inapp: true,
  },
  telegramChatId: '@prod_radar_alerts',
  emailAddresses: ['alerts@company.ru', 'manager@company.ru'],
  webhookUrl: '',
};
