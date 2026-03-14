import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { useAppStore } from '../stores/appStore';

export function AppLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const { selectedProject, searchQuery, setSearchQuery } = useAppStore();

  const activeRoute = location.pathname.replace('/', '') || 'feed';

  return (
    <div className="flex h-screen overflow-hidden bg-slate-950">
      <Sidebar
        activeRoute={activeRoute}
        onNavigate={(route) => navigate(route)}
      />
      <div className="flex-1 flex flex-col overflow-hidden">
        <TopBar
          projectName={selectedProject?.name || 'Выберите проект'}
          onProjectSwitch={() => navigate('/')}
          notificationCount={3}
          searchQuery={searchQuery}
          onSearch={setSearchQuery}
        />
        <main className="flex-1 overflow-hidden bg-slate-950">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
