import { createBrowserRouter } from 'react-router-dom';
import { ProjectList } from './pages/BrandSelect';
import { Wizard } from './pages/Wizard';
import { Feed } from './pages/Feed';
import { Analytics } from './pages/Analytics';
import { Health } from './pages/Health';
import { Settings } from './pages/Settings';
import { AppLayout } from './components/AppLayout';

export const router = createBrowserRouter([
  { path: '/', element: <ProjectList /> },
  { path: '/wizard', element: <Wizard /> },
  {
    element: <AppLayout />,
    children: [
      { path: '/feed', element: <Feed /> },
      { path: '/analytics', element: <Analytics /> },
      { path: '/health', element: <Health /> },
      { path: '/settings', element: <Settings /> },
    ],
  },
]);
