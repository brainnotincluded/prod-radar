import { useState } from 'react';
import { CheckCircle, AlertTriangle, XCircle, Clock, Zap, Filter } from 'lucide-react';
import { mockHealthStatuses, mockEvents } from '../mock';
import type { EventSeverity } from '../types';

const statusConfig = {
  healthy: { icon: CheckCircle, color: 'text-emerald-400', border: 'border-l-emerald-500', bg: 'bg-emerald-500/5', label: 'Работает' },
  degraded: { icon: AlertTriangle, color: 'text-amber-400', border: 'border-l-amber-500', bg: 'bg-amber-500/5', label: 'Деградация' },
  down: { icon: XCircle, color: 'text-red-400', border: 'border-l-red-500', bg: 'bg-red-500/5', label: 'Не работает' },
};

const severityConfig = {
  info: { color: 'text-blue-400', dot: 'bg-blue-400' },
  warning: { color: 'text-amber-400', dot: 'bg-amber-400' },
  error: { color: 'text-red-400', dot: 'bg-red-400' },
};

const typeLabels: Record<string, string> = {
  collector: 'Сбор данных',
  ml: 'ML пайплайн',
  alert: 'Уведомления',
};

export function Health() {
  const [severityFilter, setSeverityFilter] = useState<EventSeverity | 'all'>('all');

  const grouped = mockHealthStatuses.reduce<Record<string, typeof mockHealthStatuses>>((acc, s) => {
    (acc[s.type] = acc[s.type] || []).push(s);
    return acc;
  }, {});

  const filteredEvents = severityFilter === 'all'
    ? mockEvents
    : mockEvents.filter((e) => e.severity === severityFilter);

  const healthyCount = mockHealthStatuses.filter((s) => s.status === 'healthy').length;
  const totalCount = mockHealthStatuses.length;

  return (
    <div className="p-5 space-y-5 overflow-y-auto h-full">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-100 tracking-tight">Здоровье системы</h2>
        <div className="flex items-center gap-2">
          <Zap className={`w-3.5 h-3.5 ${healthyCount === totalCount ? 'text-emerald-400' : 'text-amber-400'}`} />
          <span className="text-[13px] text-slate-400">{healthyCount}/{totalCount} сервисов в норме</span>
        </div>
      </div>

      {/* Health Status Grid */}
      <div className="space-y-4">
        {Object.entries(grouped).map(([type, statuses]) => (
          <div key={type}>
            <div className="section-label mb-2">{typeLabels[type]}</div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
              {statuses.map((status) => {
                const config = statusConfig[status.status];
                const Icon = config.icon;
                return (
                  <div key={status.id} className={`card border-l-2 ${config.border} ${config.bg} p-3`}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Icon className={`w-3.5 h-3.5 ${config.color}`} />
                        <span className="text-[13px] font-medium text-slate-200">{status.name}</span>
                      </div>
                      <span className={`text-[11px] ${config.color}`}>{config.label}</span>
                    </div>
                    <div className="space-y-0.5 text-[11px] text-slate-500">
                      {status.lastFetchAt && (
                        <div className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          Последний сбор: {new Date(status.lastFetchAt).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      )}
                      {status.latency !== undefined && (
                        <div>Задержка: {status.latency}мс</div>
                      )}
                      {status.message && (
                        <div className={config.color}>{status.message}</div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Event Log */}
      <div className="card">
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/60">
          <h3 className="text-[13px] font-medium text-slate-300">Журнал событий</h3>
          <div className="flex items-center gap-1.5">
            <Filter className="w-3 h-3 text-slate-500" />
            {(['all', 'info', 'warning', 'error'] as const).map((sev) => (
              <button key={sev} onClick={() => setSeverityFilter(sev)}
                className={`px-2 py-1 text-[11px] rounded-lg transition-all duration-150 ${
                  severityFilter === sev
                    ? 'bg-blue-500/10 text-blue-400 font-medium'
                    : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'
                }`}>
                {sev === 'all' ? 'Все' : sev === 'info' ? 'Инфо' : sev === 'warning' ? 'Предупр.' : 'Ошибки'}
              </button>
            ))}
          </div>
        </div>
        <div className="divide-y divide-slate-800/40 max-h-96 overflow-y-auto">
          {filteredEvents.map((event) => {
            const config = severityConfig[event.severity];
            return (
              <div key={event.id} className="flex items-start gap-2.5 px-4 py-2.5 hover:bg-slate-800/20 transition-colors duration-150">
                <div className={`w-1.5 h-1.5 mt-1.5 rounded-full ${config.dot} shrink-0`} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-[11px] text-slate-500 font-mono">
                      {new Date(event.timestamp).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </span>
                    <span className={`text-[11px] ${config.color}`}>{event.type}</span>
                  </div>
                  <p className="text-[13px] text-slate-300 leading-relaxed">{event.description}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
