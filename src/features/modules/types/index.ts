export interface Module {
    id: string;
    name: string;
    description: string;
    icon?: string;
  }
  
  export interface ModuleState {
    modules: Module[];
    selectedModule: Module | null;
    isLoading: boolean;
    error: string | null;
  }