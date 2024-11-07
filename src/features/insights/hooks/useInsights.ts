import { useState, useEffect } from 'react';
import { InsightsService } from '../services/insightsService';

interface Insight {
  id: string;
  title: string;
  description: string;
  timestamp: Date;
}

export function useInsights() {
  const [insights, setInsights] = useState<Insight[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchInsights = async () => {
      try {
        setIsLoading(true);
        const data = await InsightsService.getInsights();
        setInsights(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch insights');
      } finally {
        setIsLoading(false);
      }
    };

    fetchInsights();
  }, []);

  return {
    insights,
    isLoading,
    error
  };
}