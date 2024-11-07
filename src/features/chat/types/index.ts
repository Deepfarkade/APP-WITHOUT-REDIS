export interface Message {
  id: string;
  text: string;
  content?: string;
  sender: 'user' | 'bot';
  role?: 'user' | 'assistant';
  timestamp: Date;
  sessionId: string;
}

export interface ChatSession {
  id: string;
  title: string;
  lastMessage?: string;
  timestamp: Date;
  messages: Message[];
  user_id: string;
}

export interface ChatState {
  sessions: ChatSession[];
  currentSession: ChatSession | null;
  isLoading: boolean;
  error: string | null;
}

export interface ChatResponse {
  id: string;
  text: string;
  sender: 'bot';
  timestamp: string;
  sessionId: string;
}