import { BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, CartesianGrid } from 'recharts';
import { TrendingUp, TrendingDown, Minus, AlertTriangle, Clock, BarChart3, MessageSquare } from 'lucide-react';
import { mockAnalytics } from '../mock';

const COLORS = ['#3b82f6', '#f97316', '#8b5cf6', '#06b6d4', '#f43f5e'];
const sourceLabels: Record<string, string> = { social: 'Соцсети', news: 'СМИ', forum: 'Форумы', blog: 'Блоги', review: 'Отзывы' };

const trendIcons = { up: TrendingUp, down: TrendingDown, stable: Minus };
const trendColors = { up: 'text-red-400', down: 'text-green-400', stable: 'text-slate-500' };

const metricAccents: Record<string, string> = {
  mentions: 'border-blue-500/30 bg-blue-500/5',
  sentiment: 'border-emerald-500/30 bg-emerald-500/5',
  spikes: 'border-orange-500/30 bg-orange-500/5',
  response: 'border-violet-500/30 bg-violet-500/5',
};

function MetricCard({ label, value, icon: Icon, color = 'text-blue-400', accent = '' }: { label: string; value: string | number; icon: React.ElementType; color?: string; accent?: string }) {
  return (
    <div className={`card border-t-2 p-4 ${accent}`}>
      <div className="flex items-center gap-1.5 mb-2">
        <Icon className={`w-3.5 h-3.5 ${color}`} />
        <span className="text-[11px] text-slate-500 uppercase tracking-wider">{label}</span>
      </div>
      <div className="text-2xl font-semibold text-slate-100 tracking-tight">{value}</div>
    </div>
  );
}

const chartTooltipStyle = {
  background: '#1e293b',
  border: '1px solid rgba(51,65,85,0.5)',
  borderRadius: 8,
  fontSize: 12,
  boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
};

export function Analytics() {
  const data = mockAnalytics;

  return (
    <div className="p-5 space-y-4 overflow-y-auto h-full">
      <h2 className="text-lg font-semibold text-slate-100 tracking-tight">Аналитика</h2>

      {/* Key Metrics */}
      <div className="grid grid-cols-4 gap-3">
        <MetricCard label="Всего упоминаний" value={data.keyMetrics.totalMentions.toLocaleString()} icon={MessageSquare} accent={metricAccents.mentions} />
        <MetricCard label="Ср. тональность" value={data.keyMetrics.avgSentiment.toFixed(2)} icon={BarChart3} color={data.keyMetrics.avgSentiment < 0 ? 'text-red-400' : 'text-green-400'} accent={data.keyMetrics.avgSentiment < 0 ? 'border-red-500/30 bg-red-500/5' : metricAccents.sentiment} />
        <MetricCard label="Всплески" value={data.keyMetrics.spikeCount} icon={AlertTriangle} color="text-orange-400" accent={metricAccents.spikes} />
        <MetricCard label="Среднее время реакции" value={`${data.keyMetrics.avgResponseTime}ч`} icon={Clock} color="text-violet-400" accent={metricAccents.response} />
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-2 gap-3">
        {/* Sentiment Over Time */}
        <div className="card p-4">
          <h3 className="text-[13px] font-medium text-slate-300 mb-3">Тональность по дням</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data.sentimentOverTime}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,0.6)" />
              <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={(d) => new Date(d).toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' })} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip contentStyle={chartTooltipStyle} />
              <Line type="monotone" dataKey="positive" stroke="#4ade80" strokeWidth={1.5} dot={false} name="Позитив" />
              <Line type="monotone" dataKey="neutral" stroke="#64748b" strokeWidth={1.5} dot={false} name="Нейтрал" />
              <Line type="monotone" dataKey="negative" stroke="#f87171" strokeWidth={1.5} dot={false} name="Негатив" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Volume */}
        <div className="card p-4">
          <h3 className="text-[13px] font-medium text-slate-300 mb-3">Объём упоминаний</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={data.volumeByDay}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,41,59,0.6)" />
              <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={(d) => new Date(d).toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' })} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip contentStyle={chartTooltipStyle} />
              <Bar dataKey="count" fill="#3b82f6" radius={[3, 3, 0, 0]} name="Упоминания" opacity={0.8} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-2 gap-3">
        {/* Source Distribution */}
        <div className="card p-4">
          <h3 className="text-[13px] font-medium text-slate-300 mb-3">Источники</h3>
          <div className="flex items-center">
            <ResponsiveContainer width="50%" height={170}>
              <PieChart>
                <Pie data={data.sourceDistribution} dataKey="count" nameKey="source" cx="50%" cy="50%" innerRadius={40} outerRadius={65} paddingAngle={3} strokeWidth={0}>
                  {data.sourceDistribution.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} opacity={0.8} />
                  ))}
                </Pie>
                <Tooltip contentStyle={chartTooltipStyle}
                  formatter={(value, name) => [String(value), sourceLabels[String(name)] || String(name)]} />
              </PieChart>
            </ResponsiveContainer>
            <div className="space-y-2.5">
              {data.sourceDistribution.map((item, i) => (
                <div key={item.source} className="flex items-center gap-2 text-[12px]">
                  <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                  <span className="text-slate-300">{sourceLabels[item.source]}</span>
                  <span className="text-slate-500 font-mono text-[11px]">{item.count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Risk Topics */}
        <div className="card p-4">
          <h3 className="text-[13px] font-medium text-slate-300 mb-3">Топ рисковых тем</h3>
          <div className="space-y-3">
            {data.topRiskTopics.map((topic) => {
              const TrendIcon = trendIcons[topic.trend];
              return (
                <div key={topic.topic} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <TrendIcon className={`w-3 h-3 ${trendColors[topic.trend]}`} />
                    <span className="text-[13px] text-slate-200">{topic.topic}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-20 h-1 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-orange-500/60 rounded-full transition-all duration-300" style={{ width: `${(topic.count / 50) * 100}%` }} />
                    </div>
                    <span className="text-[11px] font-mono text-slate-500 w-5 text-right">{topic.count}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Spike History */}
      <div className="card p-4">
        <h3 className="text-[13px] font-medium text-slate-300 mb-3">История всплесков</h3>
        <div className="space-y-0">
          {data.spikeHistory.map((spike, i) => (
            <div key={i} className="flex items-center justify-between py-2.5 border-b border-slate-800/40 last:border-0">
              <div className="flex items-center gap-2.5">
                <AlertTriangle className={`w-3.5 h-3.5 ${spike.severity === 'critical' ? 'text-red-400' : spike.severity === 'high' ? 'text-orange-400' : 'text-amber-400'}`} />
                <span className="text-[13px] text-slate-300">
                  {new Date(spike.date).toLocaleDateString('ru-RU', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-[12px] text-slate-500">{spike.mentions} упоминаний</span>
                <span className={`text-[11px] font-medium px-2 py-0.5 rounded ${
                  spike.severity === 'critical' ? 'bg-red-500/10 text-red-400 border border-red-500/15' :
                  spike.severity === 'high' ? 'bg-orange-500/10 text-orange-400 border border-orange-500/15' :
                  'bg-amber-500/10 text-amber-400 border border-amber-500/15'
                }`}>
                  {spike.severity === 'critical' ? 'Критический' : spike.severity === 'high' ? 'Высокий' : 'Средний'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
