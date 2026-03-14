import { MessageSquare, BarChart3, Activity, Settings, Radio } from 'lucide-react';

interface SidebarProps {
  activeRoute: string;
  onNavigate?: (route: string) => void;
}

const navigationItems = [
  { id: 'feed', label: 'Лента', icon: MessageSquare, route: '/feed' },
  { id: 'analytics', label: 'Аналитика', icon: BarChart3, route: '/analytics' },
  { id: 'health', label: 'Здоровье', icon: Activity, route: '/health' },
  { id: 'settings', label: 'Настройки', icon: Settings, route: '/settings' },
];

export function Sidebar({ activeRoute, onNavigate }: SidebarProps) {
  const handleClick = (route: string) => {
    if (onNavigate) {
      onNavigate(route);
    }
  };

  return (
    <aside className="w-60 h-screen bg-slate-900/50 border-r border-slate-800/60 flex flex-col">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-slate-800/60">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <Radio className="w-4 h-4 text-white" />
          </div>
          <span className="text-[15px] font-semibold text-slate-100 tracking-tight">Prod Radar</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4">
        <div className="section-label px-3 mb-2">Навигация</div>
        <div className="space-y-0.5">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeRoute === item.id || activeRoute === item.route;

            return (
              <button
                key={item.id}
                onClick={() => handleClick(item.route)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-blue-500/12 text-blue-400 shadow-[inset_2px_0_0_0] shadow-blue-500'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Icon className={`w-[18px] h-[18px] ${isActive ? 'text-blue-400' : ''}`} />
                {item.label}
              </button>
            );
          })}
        </div>
      </nav>

      {/* Footer */}
      <div className="px-5 py-3 border-t border-slate-800/60">
        <div className="text-[11px] text-slate-600">
          Prod Radar v1.0
        </div>
      </div>
    </aside>
  );
}
