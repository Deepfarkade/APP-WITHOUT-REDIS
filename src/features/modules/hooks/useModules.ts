import { useState, useEffect } from 'react';
import { Module } from '../types';
import { ModuleService } from '../services/moduleService';

export function useModules() {
  const [modules, setModules] = useState<Module[]>([]);
  const [selectedModule, setSelectedModule] = useState<Module | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModules = async () => {
      try {
        setIsLoading(true);
        const data = await ModuleService.getModules();
        setModules(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch modules');
      } finally {
        setIsLoading(false);
      }
    };

    fetchModules();
  }, []);

  const selectModule = async (moduleId: string) => {
    try {
      await ModuleService.selectModule(moduleId);
      const selected = modules.find(m => m.id === moduleId);
      setSelectedModule(selected || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to select module');
    }
  };

  return {
    modules,
    selectedModule,
    selectModule,
    isLoading,
    error
  };
}