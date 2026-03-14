import { ChevronDown, Bell, Search, User } from 'lucide-react';

interface TopBarProps {
  projectName: string;
  onProjectSwitch?: () => void;
  notificationCount?: number;
  onSearch?: (query: string) => void;
  searchQuery?: string;
}

export function TopBar({
  projectName,
  onProjectSwitch,
  notificationCount = 0,
  onSearch,
  searchQuery = ''
}: TopBarProps) {
  return (
    <header className="h-14 bg-slate-900/30 border-b border-slate-800/60 flex items-center justify-between px-5 backdrop-blur-sm">
      <button
        onClick={onProjectSwitch}
        className="flex items-center gap-2 px-2.5 py-1.5 -ml-2.5 rounded-lg hover:bg-slate-800/40 transition-colors duration-150 group"
      >
        <span className="text-sm font-semibold text-slate-100">{projectName}</span>
        <ChevronDown className="w-3.5 h-3.5 text-slate-500 group-hover:text-slate-400 transition-colors duration-150" />
      </button>

      <div className="flex-1 max-w-sm mx-6">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
          <input
            type="text"
            placeholder="Поиск по упоминаниям..."
            value={searchQuery}
            onChange={(e) => onSearch?.(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 bg-slate-800/40 border border-slate-700/40 rounded-lg text-sm text-slate-200 placeholder-slate-500 transition-all duration-150 focus:outline-none focus:bg-slate-800/60 focus:border-slate-600/60 focus:ring-1 focus:ring-blue-500/20"
          />
        </div>
      </div>

      <div className="flex items-center gap-1">
        <button className="relative p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 rounded-lg transition-all duration-150">
          <Bell className="w-[18px] h-[18px]" />
          {notificationCount > 0 && (
            <span className="absolute top-1.5 right-1.5 w-3.5 h-3.5 bg-red-500 text-white text-[10px] font-semibold rounded-full flex items-center justify-center leading-none">
              {notificationCount > 9 ? '9+' : notificationCount}
            </span>
          )}
        </button>
        <button className="p-1.5 rounded-lg hover:bg-slate-800/40 transition-all duration-150">
          <div className="w-7 h-7 bg-slate-700/60 rounded-full flex items-center justify-center border border-slate-600/40">
            <User className="w-3.5 h-3.5 text-slate-400" />
          </div>
        </button>
      </div>
    </header>
  );
}
