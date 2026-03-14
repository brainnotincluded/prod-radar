import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Bell, Send, Mail, Globe, Smartphone, TestTube, Save, Pencil } from 'lucide-react';
import { useAppStore } from '../stores/appStore';

export function Settings() {
  const navigate = useNavigate();
  const { selectedProject, alertConfig, setAlertConfig } = useAppStore();
  const [testSent, setTestSent] = useState(false);

  const handleTestAlert = () => {
    setTestSent(true);
    setTimeout(() => setTestSent(false), 2000);
  };

  return (
    <div className="p-5 space-y-4 overflow-y-auto h-full max-w-2xl">
      <h2 className="text-lg font-semibold text-slate-100 tracking-tight">Настройки</h2>

      {selectedProject && (
        <div className="card p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-[13px] font-medium text-slate-300">Проект: {selectedProject.name}</h3>
            <button onClick={() => navigate('/wizard')} className="flex items-center gap-1 text-[12px] text-blue-400 hover:text-blue-300 transition-colors duration-150">
              <Pencil className="w-3 h-3" /> Изменить
            </button>
          </div>
          <div className="grid grid-cols-3 gap-4 text-[12px]">
            <div>
              <span className="section-label block mb-1.5">Ключевые слова</span>
              <div className="flex flex-wrap gap-1">
                {selectedProject.keywords.map((k) => (
                  <span key={k} className="px-1.5 py-0.5 bg-blue-500/8 text-blue-400 rounded text-[11px] border border-blue-500/10">{k}</span>
                ))}
              </div>
            </div>
            <div>
              <span className="section-label block mb-1.5">Исключения</span>
              <div className="flex flex-wrap gap-1">
                {selectedProject.exclusions.length > 0
                  ? selectedProject.exclusions.map((e) => (
                    <span key={e} className="px-1.5 py-0.5 bg-red-500/8 text-red-400 rounded text-[11px] border border-red-500/10">{e}</span>
                  ))
                  : <span className="text-slate-600 text-[11px]">Нет</span>}
              </div>
            </div>
            <div>
              <span className="section-label block mb-1.5">Риск-слова</span>
              <div className="flex flex-wrap gap-1">
                {selectedProject.riskWords.map((r) => (
                  <span key={r} className="px-1.5 py-0.5 bg-amber-500/8 text-amber-400 rounded text-[11px] border border-amber-500/10">{r}</span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="card p-4 space-y-4">
        <div className="flex items-center gap-2">
          <Bell className="w-3.5 h-3.5 text-orange-400" />
          <h3 className="text-[13px] font-medium text-slate-300">Настройки всплесков</h3>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="section-label block mb-1.5">Порог всплеска (упоминаний в окне)</label>
            <input type="number" value={alertConfig.spikeThreshold} min={5} max={200}
              onChange={(e) => setAlertConfig({ spikeThreshold: Number(e.target.value) })}
              className="input-base" />
          </div>
          <div>
            <label className="section-label block mb-1.5">Кулдаун (минуты)</label>
            <select value={alertConfig.cooldownMinutes}
              onChange={(e) => setAlertConfig({ cooldownMinutes: Number(e.target.value) })}
              className="input-base">
              <option value={15}>15 минут</option>
              <option value={30}>30 минут</option>
              <option value={60}>1 час</option>
              <option value={120}>2 часа</option>
              <option value={240}>4 часа</option>
            </select>
          </div>
        </div>
      </div>

      <div className="card p-4 space-y-3">
        <h3 className="text-[13px] font-medium text-slate-300 mb-1">Каналы уведомлений</h3>
        <div className="space-y-2">
          <label className="flex items-center justify-between cursor-pointer py-1">
            <div className="flex items-center gap-2">
              <Send className="w-3.5 h-3.5 text-blue-400" />
              <span className="text-[13px] text-slate-200">Telegram бот</span>
            </div>
            <input type="checkbox" checked={alertConfig.channels.telegram}
              onChange={(e) => setAlertConfig({ channels: { ...alertConfig.channels, telegram: e.target.checked } })} />
          </label>
          {alertConfig.channels.telegram && (
            <input type="text" value={alertConfig.telegramChatId || ''} placeholder="Chat ID или @username"
              onChange={(e) => setAlertConfig({ telegramChatId: e.target.value })} className="input-base" />
          )}
        </div>
        <div className="border-t border-slate-800/40" />
        <div className="space-y-2">
          <label className="flex items-center justify-between cursor-pointer py-1">
            <div className="flex items-center gap-2">
              <Mail className="w-3.5 h-3.5 text-violet-400" />
              <span className="text-[13px] text-slate-200">Email</span>
            </div>
            <input type="checkbox" checked={alertConfig.channels.email}
              onChange={(e) => setAlertConfig({ channels: { ...alertConfig.channels, email: e.target.checked } })} />
          </label>
          {alertConfig.channels.email && (
            <input type="email" value={(alertConfig.emailAddresses || []).join(', ')} placeholder="email@company.ru"
              onChange={(e) => setAlertConfig({ emailAddresses: e.target.value.split(',').map((s) => s.trim()) })} className="input-base" />
          )}
        </div>
        <div className="border-t border-slate-800/40" />
        <div className="space-y-2">
          <label className="flex items-center justify-between cursor-pointer py-1">
            <div className="flex items-center gap-2">
              <Globe className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-[13px] text-slate-200">Webhook</span>
            </div>
            <input type="checkbox" checked={alertConfig.channels.webhook}
              onChange={(e) => setAlertConfig({ channels: { ...alertConfig.channels, webhook: e.target.checked } })} />
          </label>
          {alertConfig.channels.webhook && (
            <input type="url" value={alertConfig.webhookUrl || ''} placeholder="https://your-service.com/webhook"
              onChange={(e) => setAlertConfig({ webhookUrl: e.target.value })} className="input-base" />
          )}
        </div>
        <div className="border-t border-slate-800/40" />
        <label className="flex items-center justify-between cursor-pointer py-1">
          <div className="flex items-center gap-2">
            <Smartphone className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-[13px] text-slate-200">Push в приложении</span>
          </div>
          <input type="checkbox" checked={alertConfig.channels.inapp}
            onChange={(e) => setAlertConfig({ channels: { ...alertConfig.channels, inapp: e.target.checked } })} />
        </label>
      </div>

      <div className="flex items-center gap-2 pt-1">
        <button onClick={handleTestAlert}
          className="flex items-center gap-1.5 px-3 py-2 text-[13px] bg-orange-500/8 text-orange-400 border border-orange-500/15 rounded-lg hover:bg-orange-500/15 transition-all duration-150">
          <TestTube className="w-3.5 h-3.5" />
          {testSent ? 'Отправлено!' : 'Тестовый алерт'}
        </button>
        <button className="flex items-center gap-1.5 px-4 py-2 text-[13px] font-medium bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-all duration-150">
          <Save className="w-3.5 h-3.5" /> Сохранить
        </button>
      </div>
    </div>
  );
}
