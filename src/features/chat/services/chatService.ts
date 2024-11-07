import { api } from '../../../lib/api';
import { Message, ChatSession } from '../types';

export class ChatService {
  static async sendMessage(content: string, sessionId: string): Promise<Message> {
    try {
      const response = await api.post<Message>(`/chat/${sessionId}/send`, { text: content });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to send message');
    }
  }

  static async getChatHistory(): Promise<ChatSession[]> {
    try {
      const response = await api.get<ChatSession[]>('/chat/sessions');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to fetch chat history');
    }
  }

  static async createNewChat(): Promise<ChatSession> {
    try {
      const response = await api.post<ChatSession>('/chat/sessions');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to create new chat');
    }
  }

  static async getSessionMessages(sessionId: string): Promise<Message[]> {
    try {
      const response = await api.get<Message[]>(`/chat/${sessionId}/messages`);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to fetch session messages');
    }
  }
}