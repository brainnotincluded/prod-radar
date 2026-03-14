import { create } from 'zustand';
import type { Project, ProjectConfig, AlertConfig, FilterState } from '../types';
import { mockProjects } from '../mock';
import { defaultAlertConfig } from '../mock/alerts';

interface AppState {
  projects: Project[];
  selectedProject: Project | null;
  alertConfig: AlertConfig;
  filters: FilterState;
  searchQuery: string;

  selectProject: (project: Project) => void;
  createProject: (name: string) => Project;
  saveProjectConfig: (projectId: string, config: ProjectConfig) => void;
  setFilters: (filters: Partial<FilterState>) => void;
  setSearchQuery: (query: string) => void;
  setAlertConfig: (config: Partial<AlertConfig>) => void;
}

export const useAppStore = create<AppState>((set) => ({
  projects: mockProjects,
  selectedProject: null,
  alertConfig: defaultAlertConfig,
  filters: {
    sentiments: [],
    sources: [],
    riskLevels: [],
    minRelevance: 0,
    timeRange: 'week',
  },
  searchQuery: '',

  selectProject: (project) => set({ selectedProject: project }),

  createProject: (name) => {
    const newProject: Project = {
      id: `proj-${Date.now()}`,
      name,
      keywords: [],
      exclusions: [],
      riskWords: [],
      createdAt: new Date(),
    };
    set((state) => ({ projects: [...state.projects, newProject], selectedProject: newProject }));
    return newProject;
  },

  saveProjectConfig: (projectId, config) =>
    set((state) => ({
      projects: state.projects.map((p) =>
        p.id === projectId ? { ...p, ...config } : p
      ),
      selectedProject: state.selectedProject?.id === projectId
        ? { ...state.selectedProject, ...config }
        : state.selectedProject,
    })),

  setFilters: (filters) =>
    set((state) => ({ filters: { ...state.filters, ...filters } })),

  setSearchQuery: (searchQuery) => set({ searchQuery }),

  setAlertConfig: (config) =>
    set((state) => ({ alertConfig: { ...state.alertConfig, ...config } })),
}));
