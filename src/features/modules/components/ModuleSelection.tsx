import React from 'react';
import { useModules } from '../hooks/useModules';

export default function ModuleSelection() {
  const { modules, selectModule, selectedModule } = useModules();

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {modules.map(module => (
        <button
          key={module.id}
          onClick={() => selectModule(module.id)}
          className={`p-4 rounded-lg ${
            selectedModule?.id === module.id
              ? 'bg-indigo-600 text-white'
              : 'bg-white hover:bg-gray-50'
          }`}
        >
          <h3 className="text-lg font-semibold">{module.name}</h3>
          <p className="text-sm">{module.description}</p>
        </button>
      ))}
    </div>
  );
}