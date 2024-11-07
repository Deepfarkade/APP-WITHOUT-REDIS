import { api } from '../../../lib/api';

export class InsightsService {
  static async getInsights() {
    const response = await api.get('/insights');
    return response.data;
  }

  static async generateSummary(data: any) {
    const response = await api.post('/insights/summary', data);
    return response.data;
  }
}