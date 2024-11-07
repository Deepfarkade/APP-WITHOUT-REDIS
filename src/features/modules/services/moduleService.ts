import { api } from '../../../lib/api';
import { Module } from '../types';

export class ModuleService {
  static async getModules(): Promise<Module[]> {
    const response = await api.get('/modules');
    return response.data;
  }

  static async selectModule(moduleId: string): Promise<void> {
    await api.post(`/modules/${moduleId}/select`);
  }
}