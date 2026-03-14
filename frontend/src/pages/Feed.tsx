import { useState, useMemo } from 'react';
import { ChevronDown, ChevronUp, ExternalLink, Users, Clock, Layers, Newspaper, MessageCircle, BookOpen, MessagesSquare, Star } from 'lucide-react';
import { Badge } from '../components/Badge';
import { useAppStore } from '../stores/appStore';
import { mockMentions } from '../mock';
import type { Mention, Sentiment, MentionSource } from '../types';

const sourceIcons: Record<string, React.ElementType> = {
  news: Newspaper, social: MessageCircle, blog: BookOpen, forum: MessagesSquare, review: Star,
};

const sourceLabels: Record<string, string> = {
  news: 'СМИ', social: 'Соцсети', blog: 'Блоги', forum: 'Форумы', review: 'Отзывы',
};

const timeRangeLabels: Record<string, string> = {
  today: 'Сегодня', '3days': '3 дня', week: 'Неделя', month: 'Месяц',
};

const riskBorderColors: Record<string, string> = {
  critical: 'border-l-red-500',
  high: 'border-l-orange-500',
  medium: 'border-l-amber-500',
  low: 'border-l-blue-500',
  none: 'border-l-transparent',
};

function MentionCard({ mention }: { mention: Mention }) {
  const [expanded, setExpanded] = useState(false);
  const SourceIcon = sourceIcons[mention.source] || Newspaper;

  return (
    <div className={`card border-l-2 ${riskBorderColors[mention.riskLevel]} p-4 hover:bg-slate-800/30 transition-all duration-150`}>
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          {/* Meta row */}
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <span className="inline-flex items-center gap-1 text-[11px] text-slate-500">
              <SourceIcon className="w-3 h-3" />
              {mention.sourceName}
            </span>
            <Badge type="sentiment" value={mention.sentiment} />
            {mention.riskLevel !== 'none' && <Badge type="risk" value={mention.riskLevel} />}
            {mention.clusterSize && mention.clusterSize > 1 && (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] bg-slate-800/60 text-slate-400 border border-slate-700/40">
                <Layers className="w-3 h-3" /> {mention.clusterSize} похожих
              </span>
            )}
          </div>

          {/* Title */}
          <h3 className="text-[13px] font-semibold text-slate-100 leading-snug mb-1">{mention.title}</h3>
          <p className="text-[12px] text-slate-400 line-clamp-2 leading-relaxed">{mention.content}</p>
        </div>

        {/* Right column */}
        <div className="text-right shrink-0 space-y-2">
          <div className="text-[11px] text-slate-500 flex items-center gap-1 justify-end">
            <Clock className="w-3 h-3" />
            {new Date(mention.publishedAt).toLocaleDateString('ru-RU', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}
          </div>
          <div>
            <div className="text-[11px] text-slate-500 mb-1">Релевантность</div>
            <div className="flex items-center gap-1.5">
              <div className="w-14 h-1 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full rounded-full bg-blue-500/70 transition-all duration-300" style={{ width: `${mention.relevance}%` }} />
              </div>
              <span className="text-[11px] font-mono text-slate-400">{mention.relevance}%</span>
            </div>
          </div>
          {mention.reach > 0 && (
            <div className="flex items-center gap-1 text-[11px] text-slate-500 justify-end">
              <Users className="w-3 h-3" /> {mention.reach.toLocaleString('ru-RU')}
            </div>
          )}
        </div>
      </div>

      {/* Expandable details */}
      <button onClick={() => setExpanded(!expanded)} className="mt-3 text-[12px] text-slate-500 hover:text-slate-300 flex items-center gap-1 transition-colors duration-150">
        {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        {expanded ? 'Скрыть детали' : 'Почему релевантно?'}
      </button>
      {expanded && (
        <div className="mt-2 p-3 bg-slate-800/30 rounded-lg border border-slate-700/30 space-y-1.5 text-[12px]">
          <div className="flex gap-2">
            <span className="text-slate-500 shrink-0">Совпавшие слова:</span>
            <span className="text-slate-300">{mention.highlights.join(', ')}</span>
          </div>
          <div className="flex gap-2">
            <span className="text-slate-500 shrink-0">ML уверенность:</span>
            <span className="text-slate-300">{(mention.mlConfidence * 100).toFixed(0)}%</span>
          </div>
          <div className="flex gap-2">
            <span className="text-slate-500 shrink-0">Приоритет:</span>
            <span className={mention.riskLevel === 'critical' ? 'text-red-400' : mention.riskLevel === 'high' ? 'text-orange-400' : 'text-slate-300'}>
              {mention.riskLevel === 'critical' ? 'Критический' : mention.riskLevel === 'high' ? 'Высокий' : mention.riskLevel === 'medium' ? 'Средний' : mention.riskLevel === 'low' ? 'Низкий' : 'Обычный'}
            </span>
          </div>
          <a href={mention.sourceUrl} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300 transition-colors duration-150 pt-1">
            Открыть источник <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      )}
    </div>
  );
}

export function Feed() {
  const { filters, setFilters, searchQuery } = useAppStore();
  const [sortBy, setSortBy] = useState<'priority' | 'time' | 'relevance'>('priority');

  const riskOrder = { critical: 0, high: 1, medium: 2, low: 3, none: 4 };

  const filtered = useMemo(() => {
    let result = [...mockMentions];

    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      result = result.filter((m) => m.title.toLowerCase().includes(q) || m.content.toLowerCase().includes(q));
    }
    if (filters.sentiments.length > 0) {
      result = result.filter((m) => filters.sentiments.includes(m.sentiment));
    }
    if (filters.sources.length > 0) {
      result = result.filter((m) => filters.sources.includes(m.source));
    }
    if (filters.riskLevels.length > 0) {
      result = result.filter((m) => filters.riskLevels.includes(m.riskLevel));
    }
    if (filters.minRelevance > 0) {
      result = result.filter((m) => m.relevance >= filters.minRelevance);
    }

    if (sortBy === 'priority') {
      result.sort((a, b) => riskOrder[a.riskLevel] - riskOrder[b.riskLevel] || b.relevance - a.relevance);
    } else if (sortBy === 'time') {
      result.sort((a, b) => new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime());
    } else {
      result.sort((a, b) => b.relevance - a.relevance);
    }
    return result;
  }, [searchQuery, filters, sortBy]);

  const negCount = filtered.filter((m) => m.sentiment === 'negative').length;
  const spikeCount = filtered.filter((m) => m.riskLevel === 'critical' || m.riskLevel === 'high').length;

  const toggleSentiment = (s: Sentiment) => {
    const current = filters.sentiments;
    setFilters({ sentiments: current.includes(s) ? current.filter((x) => x !== s) : [...current, s] });
  };

  const toggleSource = (s: MentionSource) => {
    const current = filters.sources;
    setFilters({ sources: current.includes(s) ? current.filter((x) => x !== s) : [...current, s] });
  };

  return (
    <div className="flex h-full">
      {/* Sidebar Filters */}
      <aside className="w-52 shrink-0 border-r border-slate-800/60 py-4 overflow-y-auto">
        {/* Sentiment */}
        <div className="px-4 pb-4 border-b border-slate-800/40">
          <div className="section-label mb-2.5">Тональность</div>
          <div className="space-y-1">
            {(['positive', 'neutral', 'negative'] as Sentiment[]).map((s) => (
              <label key={s} className="flex items-center gap-2 py-1 cursor-pointer hover:bg-slate-800/30 -mx-1 px-1 rounded transition-colors duration-150">
                <input type="checkbox" checked={filters.sentiments.includes(s)} onChange={() => toggleSentiment(s)} />
                <Badge type="sentiment" value={s} />
              </label>
            ))}
          </div>
        </div>

        {/* Source */}
        <div className="px-4 py-4 border-b border-slate-800/40">
          <div className="section-label mb-2.5">Источник</div>
          <div className="space-y-1">
            {(['news', 'social', 'blog', 'forum', 'review'] as MentionSource[]).map((s) => {
              const Icon = sourceIcons[s];
              return (
                <label key={s} className="flex items-center gap-2 py-1 cursor-pointer text-[13px] text-slate-300 hover:bg-slate-800/30 -mx-1 px-1 rounded transition-colors duration-150">
                  <input type="checkbox" checked={filters.sources.includes(s)} onChange={() => toggleSource(s)} />
                  <Icon className="w-3.5 h-3.5 text-slate-500" />
                  {sourceLabels[s]}
                </label>
              );
            })}
          </div>
        </div>

        {/* Relevance */}
        <div className="px-4 py-4 border-b border-slate-800/40">
          <div className="section-label mb-2.5">Релевантность: {filters.minRelevance}%+</div>
          <input type="range" min={0} max={100} step={5} value={filters.minRelevance}
            onChange={(e) => setFilters({ minRelevance: Number(e.target.value) })} />
        </div>

        {/* Time Range */}
        <div className="px-4 pt-4">
          <div className="section-label mb-2.5">Период</div>
          <div className="space-y-0.5">
            {Object.entries(timeRangeLabels).map(([key, label]) => (
              <button key={key} onClick={() => setFilters({ timeRange: key as typeof filters.timeRange })}
                className={`block w-full text-left px-2 py-1.5 text-[13px] rounded-lg transition-all duration-150 ${
                  filters.timeRange === key
                    ? 'bg-blue-500/10 text-blue-400 font-medium'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/30'
                }`}>
                {label}
              </button>
            ))}
          </div>
        </div>
      </aside>

      {/* Main Feed */}
      <div className="flex-1 p-5 overflow-y-auto">
        {/* Stats Bar */}
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800/40">
          <div className="flex items-center gap-4 text-[13px]">
            <span className="text-slate-400 font-medium">{filtered.length} упоминаний</span>
            {negCount > 0 && (
              <span className="text-red-400/80">{negCount} негативных</span>
            )}
            {spikeCount > 0 && (
              <span className="text-orange-400/80">{spikeCount} всплесков</span>
            )}
          </div>
          <div className="flex items-center gap-1 text-[12px]">
            <span className="text-slate-500 mr-1">Сортировка:</span>
            {[
              { key: 'priority', label: 'Приоритет' },
              { key: 'time', label: 'Время' },
              { key: 'relevance', label: 'Релевантность' },
            ].map(({ key, label }) => (
              <button key={key} onClick={() => setSortBy(key as typeof sortBy)}
                className={`px-2 py-1 rounded-lg transition-all duration-150 ${
                  sortBy === key
                    ? 'bg-blue-500/10 text-blue-400 font-medium'
                    : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'
                }`}>
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Mention List */}
        <div className="space-y-2">
          {filtered.map((mention) => (
            <MentionCard key={mention.id} mention={mention} />
          ))}
          {filtered.length === 0 && (
            <div className="text-center py-20 text-slate-500 text-sm">
              Нет упоминаний по заданным фильтрам
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
