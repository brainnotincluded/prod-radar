import type { Project } from '../types';

export const mockProjects: Project[] = [
  {
    id: 'proj-1',
    name: 'Мониторинг Т-Банка',
    keywords: ['Т-Банк', 'ТБанк', 'Tinkoff', 'Тинькофф', 'Т-Банк бизнес'],
    exclusions: ['тинькофф джаз', 'тинькофф арена'],
    riskWords: ['взлом', 'мошенничество', 'сбой', 'жалоба', 'штраф', 'претензия'],
    createdAt: new Date('2024-02-15'),
  },
  {
    id: 'proj-2',
    name: 'Яндекс PR',
    keywords: ['Яндекс', 'Yandex', 'Яндекс Такси', 'Яндекс Еда'],
    exclusions: ['яндекс погода'],
    riskWords: ['утечка данных', 'сбой сервиса', 'жалоба'],
    createdAt: new Date('2024-03-01'),
  },
];
