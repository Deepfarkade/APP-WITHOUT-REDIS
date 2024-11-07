import { useState, useCallback, useEffect } from 'react';
import { Message, ChatSession } from '../types';
import { ChatService } from '../services/chatService';
import { useAuth } from '../../auth/context/AuthContext';

function generateNewId(): string {
    return Math.random().toString(36).substr(2, 9); // Example ID generation
}

export function useChat() {
  const { user } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (user) {
      const fetchHistory = async () => {
        try {
          setIsLoading(true);
          const sessions = await ChatService.getChatHistory();
          if (sessions.length > 0) {
            setCurrentSession(sessions[0]);
            setMessages(sessions[0].messages);
          }
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to load chat history');
        } finally {
          setIsLoading(false);
        }
      };

      fetchHistory();
    }
  }, [user]);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading || !currentSession) return;

    try {
      setIsLoading(true);
      setError(null);

      const userMessage: Message = {
        id: Date.now().toString(),
        text: content.trim(),
        sender: 'user',
        role: 'user',
        timestamp: new Date(),
        sessionId: currentSession.id
      };

      setMessages(prev => [...prev, userMessage]);

      const response = await ChatService.sendMessage(content, currentSession.id);
      
      const botMessage: Message = {
        id: response.id,
        text: response.text,
        sender: 'bot',
        role: 'assistant',
        timestamp: new Date(response.timestamp),
        sessionId: currentSession.id
      };
      
      setMessages(prev => [...prev, botMessage]);
      
      if (currentSession) {
        setCurrentSession((prev: ChatSession | null) => ({
          lastMessage: botMessage.text,
          messages: [...(prev?.messages || []), userMessage, botMessage],
          id: prev?.id || generateNewId(),
          title: prev?.title || "New Session",
          timestamp: new Date(),
          user_id: prev?.user_id || "default_user_id"
        }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, currentSession]);

  return {
    messages,
    currentSession,
    isLoading,
    error,
    sendMessage
  };
}