import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, ArrowRight, Check, Search, Shield, AlertTriangle } from 'lucide-react';
import { TagInput } from '../components/TagInput';
import { useAppStore } from '../stores/appStore';

const steps = [
  { id: 0, title: 'Ключевые слова', description: 'Слова и фразы для поиска упоминаний', icon: Search },
  { id: 1, title: 'Исключения', description: 'Слова для фильтрации нерелевантного шума', icon: Shield },
  { id: 2, title: 'Риск-слова', description: 'Слова, повышающие приоритет упоминания', icon: AlertTriangle },
];

const genericSuggestions: Record<number, string[]> = {
  1: ['промокод', 'реклама', 'партнёрская ссылка', 'розыгрыш призов', 'спам'],
  2: ['утечка', 'суд', 'мошенничество', 'сбой', 'штраф', 'жалоба', 'блокировка'],
};

export function Wizard() {
  const navigate = useNavigate();
  const { selectedProject, saveProjectConfig } = useAppStore();
  const [step, setStep] = useState(0);
  const [suggesting, setSuggesting] = useState(false);

  const [keywords, setKeywords] = useState<string[]>(selectedProject?.keywords || []);
  const [exclusions, setExclusions] = useState<string[]>(selectedProject?.exclusions || []);
  const [riskWords, setRiskWords] = useState<string[]>(selectedProject?.riskWords || []);

  if (!selectedProject) {
    navigate('/');
    return null;
  }

  const currentTags = [keywords, exclusions, riskWords][step];
  const setCurrentTags = [setKeywords, setExclusions, setRiskWords][step];

  const handleSuggest = () => {
    setSuggesting(true);
    setTimeout(() => {
      if (step === 0) {
        // For keywords, generate variations from existing keywords
        const variations = keywords.flatMap((k) => {
          const lower = k.toLowerCase();
          const results: string[] = [];
          if (/[а-я]/.test(lower)) results.push(k.replace(/\s/g, ''));
          if (/[a-z]/.test(lower)) results.push(k.toUpperCase());
          return results;
        }).filter((s) => !currentTags.includes(s));
        if (variations.length > 0) {
          setCurrentTags([...currentTags, ...variations.slice(0, 3)]);
        }
      } else {
        const suggestions = genericSuggestions[step] || [];
        const newTags = suggestions.filter((s) => !currentTags.includes(s));
        setCurrentTags([...currentTags, ...newTags]);
      }
      setSuggesting(false);
    }, 1200);
  };

  const handleFinish = () => {
    saveProjectConfig(selectedProject.id, { keywords, exclusions, riskWords });
    navigate('/feed');
  };

  const canProceed = step === 0 ? keywords.length > 0 : true;

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-8">
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900/50 via-slate-950 to-slate-950 pointer-events-none" />

      <div className="w-full max-w-xl relative z-10">
        <div className="mb-8">
          <button onClick={() => navigate('/')} className="text-slate-500 hover:text-slate-300 text-[13px] mb-4 flex items-center gap-1 transition-colors duration-150">
            <ArrowLeft className="w-3.5 h-3.5" /> Назад к проектам
          </button>
          <h1 className="text-xl font-semibold text-slate-100 tracking-tight">Настройка: {selectedProject.name}</h1>
          <p className="text-slate-500 text-[13px] mt-1">Шаг {step + 1} из 3</p>
        </div>

        <div className="flex gap-2 mb-8">
          {steps.map((s) => (
            <div key={s.id} className="flex-1">
              <div className={`h-1 rounded-full transition-all duration-300 ${s.id <= step ? 'bg-blue-500' : 'bg-slate-800/60'}`} />
              <div className="flex items-center gap-1.5 mt-2.5">
                <div className={`w-5 h-5 rounded flex items-center justify-center transition-colors duration-200 ${s.id <= step ? 'bg-blue-500/10' : 'bg-slate-800/30'}`}>
                  <s.icon className={`w-3 h-3 ${s.id <= step ? 'text-blue-400' : 'text-slate-600'}`} />
                </div>
                <span className={`text-[11px] font-medium ${s.id <= step ? 'text-slate-300' : 'text-slate-600'}`}>{s.title}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="card p-5">
          <h2 className="text-[15px] font-semibold text-slate-100 mb-1">{steps[step].title}</h2>
          <p className="text-slate-500 text-[13px] mb-4">{steps[step].description}</p>
          <TagInput
            tags={currentTags}
            onChange={setCurrentTags}
            placeholder={step === 0 ? 'Введите ключевое слово...' : step === 1 ? 'Введите слово-исключение...' : 'Введите риск-слово...'}
            onSuggest={handleSuggest}
            isSuggesting={suggesting}
          />
          {step === 0 && keywords.length === 0 && (
            <p className="text-amber-400/70 text-[12px] mt-2">Добавьте хотя бы одно ключевое слово</p>
          )}
        </div>

        <div className="flex justify-between mt-5">
          <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}
            className="flex items-center gap-1.5 px-3 py-2 text-[13px] text-slate-400 hover:text-slate-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors duration-150">
            <ArrowLeft className="w-3.5 h-3.5" /> Назад
          </button>
          {step < 2 ? (
            <button onClick={() => setStep(step + 1)} disabled={!canProceed}
              className="flex items-center gap-1.5 px-4 py-2 text-[13px] font-medium bg-blue-600 text-white rounded-lg hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-150">
              Далее <ArrowRight className="w-3.5 h-3.5" />
            </button>
          ) : (
            <button onClick={handleFinish} disabled={!canProceed}
              className="flex items-center gap-1.5 px-4 py-2 text-[13px] font-medium bg-emerald-600 text-white rounded-lg hover:bg-emerald-500 disabled:opacity-40 transition-all duration-150">
              <Check className="w-3.5 h-3.5" /> Готово
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
