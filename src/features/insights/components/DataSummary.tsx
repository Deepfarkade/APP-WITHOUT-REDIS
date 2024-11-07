import React from 'react';
import { useInsights } from '../hooks/useInsights';

export default function DataSummary() {
  const { insights, isLoading } = useInsights();

  if (isLoading) {
    return <div>Loading insights...</div>;
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4">Data Summary & Insights</h2>
      {/* Render insights data */}
    </div>
  );
}