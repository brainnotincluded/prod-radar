import { AlertCircle, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import type { Sentiment, RiskLevel } from '../types';

interface BadgeProps {
  type: 'sentiment' | 'risk';
  value: Sentiment | RiskLevel;
}

const sentimentConfig = {
  positive: {
    label: 'Позитивная',
    className: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  },
  negative: {
    label: 'Негативная',
    className: 'bg-red-500/10 text-red-400 border-red-500/20',
  },
  neutral: {
    label: 'Нейтральная',
    className: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
  },
};

const riskConfig = {
  critical: {
    label: 'Критический',
    className: 'bg-red-500/10 text-red-400 border-red-500/20',
    icon: AlertCircle,
  },
  high: {
    label: 'Высокий',
    className: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
    icon: AlertTriangle,
  },
  medium: {
    label: 'Средний',
    className: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    icon: AlertTriangle,
  },
  low: {
    label: 'Низкий',
    className: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    icon: Info,
  },
  none: {
    label: 'Нет',
    className: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
    icon: CheckCircle,
  },
};

export function Badge({ type, value }: BadgeProps) {
  if (type === 'sentiment') {
    const config = sentimentConfig[value as Sentiment];
    return (
      <span
        className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium border ${config.className}`}
      >
        {config.label}
      </span>
    );
  }

  const config = riskConfig[value as RiskLevel];
  const Icon = config.icon;

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium border ${config.className}`}
    >
      <Icon className="w-3 h-3" />
      {config.label}
    </span>
  );
}
