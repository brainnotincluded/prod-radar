import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, FolderOpen, ChevronRight, Radio } from 'lucide-react';
import { useAppStore } from '../stores/appStore';

export function ProjectList() {
  const navigate = useNavigate();
  const { projects, selectProject, createProject } = useAppStore();
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');

  const handleProjectClick = (project: typeof projects[0]) => {
    selectProject(project);
    if (project.keywords.length > 0) {
      navigate('/feed');
    } else {
      navigate('/wizard');
    }
  };

  const handleCreate = () => {
    if (!newName.trim()) return;
    createProject(newName.trim());
    setNewName('');
    setShowCreate(false);
    navigate('/wizard');
  };

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-8">
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900/50 via-slate-950 to-slate-950 pointer-events-none" />

      <div className="relative z-10 w-full max-w-xl">
        <div className="text-center mb-10">
          <div className="flex items-center justify-center gap-2.5 mb-4">
            <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
              <Radio className="w-5 h-5 text-white" />
            </div>
          </div>
          <h1 className="text-2xl font-semibold text-slate-100 mb-1.5 tracking-tight">Prod Radar</h1>
          <p className="text-slate-500 text-sm">Выберите проект или создайте новый</p>
        </div>

        {/* Project List */}
        <div className="space-y-2 mb-4">
          {projects.map((project) => (
            <button
              key={project.id}
              onClick={() => handleProjectClick(project)}
              className="group w-full card p-4 text-left hover:border-slate-700/60 hover:bg-slate-800/20 transition-all duration-150 flex items-center gap-3"
            >
              <div className="w-9 h-9 rounded-lg bg-blue-500/10 border border-blue-500/15 flex items-center justify-center shrink-0">
                <FolderOpen className="w-4 h-4 text-blue-400" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-slate-100 text-[13px] font-semibold">{project.name}</h3>
                <p className="text-slate-500 text-[11px] mt-0.5">
                  {project.keywords.length > 0
                    ? `${project.keywords.length} ключевых слов`
                    : 'Не настроен'}
                </p>
              </div>
              {project.keywords.length > 0 && (
                <span className="w-1.5 h-1.5 bg-green-500/80 rounded-full shrink-0" />
              )}
              <ChevronRight className="w-3.5 h-3.5 text-slate-700 group-hover:text-slate-500 transition-colors duration-150 shrink-0" />
            </button>
          ))}
        </div>

        {/* Create Project */}
        {showCreate ? (
          <div className="card p-4">
            <label className="section-label block mb-2">Название проекта</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
                placeholder="Например: Мониторинг Сбера"
                autoFocus
                className="input-base flex-1"
              />
              <button
                onClick={handleCreate}
                disabled={!newName.trim()}
                className="px-4 py-2 text-[13px] font-medium bg-blue-600 text-white rounded-lg hover:bg-blue-500 disabled:opacity-40 transition-all duration-150"
              >
                Создать
              </button>
              <button
                onClick={() => { setShowCreate(false); setNewName(''); }}
                className="px-3 py-2 text-[13px] text-slate-400 hover:text-slate-200 transition-colors duration-150"
              >
                Отмена
              </button>
            </div>
          </div>
        ) : (
          <button
            onClick={() => setShowCreate(true)}
            className="w-full card p-4 text-left hover:border-slate-700/60 hover:bg-slate-800/20 transition-all duration-150 flex items-center gap-3 border-dashed"
          >
            <div className="w-9 h-9 rounded-lg bg-slate-800/50 border border-slate-700/30 flex items-center justify-center">
              <Plus className="w-4 h-4 text-slate-400" />
            </div>
            <span className="text-slate-400 text-[13px]">Создать новый проект</span>
          </button>
        )}
      </div>
    </div>
  );
}
