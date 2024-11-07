import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Send, Bot, Plus, MessageSquare, LogOut, ChevronLeft, ChevronRight, Download, Home, Sun, Moon } from 'lucide-react';
import { useAuth } from '../../auth/context/AuthContext';
import { useNavigate } from 'react-router-dom';
import { ChatService } from '../services/chatService';
import { Message, ChatSession } from '../types';
import UserAvatar from '../../../components/UserAvatar';
import Tooltip from '../../../components/Tooltip';

interface ChatWindowProps {
  initialSession?: ChatSession;
}

export default function ChatWindow({ initialSession }: ChatWindowProps) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(initialSession || null);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [searchQuery, setSearchQuery] = useState('');

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const history = await ChatService.getChatHistory();
        setSessions(history);
        if (!currentSession && history.length > 0) {
          setCurrentSession(history[0]);
        } else if (!history.length) {
          handleNewChat();
        }
      } catch (error) {
        console.error('Failed to fetch chat history:', error);
      }
    };

    fetchSessions();
  }, [currentSession]);

  useEffect(() => {
    scrollToBottom();
  }, [currentSession?.messages, scrollToBottom]);




  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
    document.documentElement.classList.toggle('dark');
  };

  const handleNewChat = async () => {
    try {
      setIsLoading(true);
      const newSession = await ChatService.createNewChat();
      setSessions(prev => [newSession, ...prev]);
      setCurrentSession(newSession);
      setInputMessage('');
    } catch (error) {
      console.error('Failed to create new chat:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading || !currentSession) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputMessage.trim(),
      sender: 'user',
      role: 'user',
      timestamp: new Date(),
      sessionId: currentSession.id
    };

    setCurrentSession((prev: ChatSession | null) => {
      if (!prev) return null;
      return {
        ...prev,
        messages: [...prev.messages, userMessage]
      };
    });
    
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await ChatService.sendMessage(inputMessage.trim(), currentSession.id);
      
      setCurrentSession((prev: ChatSession | null) => {
        if (!prev) return null;
        return {
          ...prev,
          messages: [...prev.messages, response],
          lastMessage: response.text,
          timestamp: new Date()
        };
      });

      setSessions(prev => prev.map(session => 
        session.id === currentSession.id 
          ? { ...session, lastMessage: response.text, timestamp: new Date() }
          : session
      ));
    } catch (error) {
      console.error('Failed to send message:', error);
      setCurrentSession((prev: ChatSession | null) => {
        if (!prev) return null;
        return {
          ...prev,
          messages: [...prev.messages, {
            id: Date.now().toString(),
            text: 'Sorry, I encountered an error. Please try again.',
            sender: 'bot',
            role: 'assistant',
            timestamp: new Date(),
            sessionId: currentSession.id
          }]
        };
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSessionSelect = async (session: ChatSession) => {
    try {
      setIsLoading(true);
      const messages = await ChatService.getSessionMessages(session.id);
      setCurrentSession({ ...session, messages });
    } catch (error) {
      console.error('Failed to load session:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const filteredSessions = sessions.filter(session => 
    session.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    session.lastMessage?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className={`flex h-screen ${theme === 'dark' ? 'bg-dark-bg' : 'bg-light-bg'}`}>
      <div className={`${
        isSidebarOpen ? 'w-64' : 'w-20'
      } ${
        theme === 'dark' ? 'bg-dark-sidebar' : 'bg-light-sidebar'
      } border-r ${
        theme === 'dark' ? 'border-dark-border' : 'border-light-border'
      } transition-all duration-300 flex flex-col shadow-lg`}>
        <div className="p-4 space-y-2">
          <button
            onClick={handleNewChat}
            className="flex items-center justify-center w-full px-4 py-3 rounded-lg bg-primary text-secondary-dark hover:bg-primary-dark transition-all duration-300 transform hover:scale-105 shadow-light-glow hover:shadow-xl group"
          >
            {isSidebarOpen ? (
              <>
                <Plus className="w-5 h-5 mr-2" />
                <span>New Analysis</span>
              </>
            ) : (
              <Plus className="w-5 h-5" />
            )}
          </button>

          <button
            onClick={() => navigate('/')}
            className={`flex items-center justify-center w-full px-4 py-3 rounded-lg ${
              theme === 'dark' 
                ? 'bg-dark-hover text-primary hover:bg-dark-hover/80' 
                : 'bg-light-hover text-primary hover:bg-light-hover/80'
            } transition-all duration-300 transform hover:scale-105 shadow-md hover:shadow-lg group`}
          >
            {isSidebarOpen ? (
              <>
                <Home className="w-5 h-5 mr-2" />
                <span>Home</span>
              </>
            ) : (
              <Home className="w-5 h-5" />
            )}
          </button>

          {isSidebarOpen && (
            <input
              type="text"
              placeholder="Search chats..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={`w-full px-3 py-2 rounded-lg ${
                theme === 'dark'
                  ? 'bg-dark-hover text-dark-text-primary placeholder-dark-text-secondary'
                  : 'bg-light-hover text-light-text-primary placeholder-light-text-secondary border border-light-border'
              } focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-300 shadow-inner`}
            />
          )}
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-2">
          {filteredSessions.map((session) => (
            <button
              key={session.id}
              onClick={() => handleSessionSelect(session)}
              className={`w-full text-left mb-2 p-3 rounded-lg hover:bg-gray-800 transition-all duration-300 transform hover:scale-105 ${
                currentSession?.id === session.id 
                  ? theme === 'dark' ? 'bg-dark-hover' : 'bg-light-hover' 
                  : ''
              }`}
            >
              <div className="flex items-center">
                <MessageSquare className={`w-4 h-4 ${
                  currentSession?.id === session.id ? 'text-primary' : 'text-gray-400'
                } group-hover:text-primary mr-2`} />
                {isSidebarOpen && (
                  <div className="overflow-hidden">
                    <p className={`text-sm ${
                      theme === 'dark' ? 'text-dark-text-primary' : 'text-light-text-primary'
                    } truncate`}>
                      {session.title || session.lastMessage}
                    </p>
                    <p className={`text-xs ${
                      theme === 'dark' ? 'text-dark-text-secondary' : 'text-light-text-secondary'
                    }`}>
                      {new Date(session.timestamp).toLocaleDateString()}
                    </p>
                  </div>
                )}
              </div>
            </button>
          ))}
        </div>

        <div className={`p-4 border-t ${
          theme === 'dark' ? 'border-dark-border' : 'border-light-border'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <Tooltip content={`${user?.full_name} (${user?.email})`}>
              <div className="flex items-center space-x-2">
                <UserAvatar name={user?.full_name || ''} size="sm" />
                {isSidebarOpen && (
                  <span className={`text-sm ${
                    theme === 'dark' ? 'text-dark-text-primary' : 'text-light-text-primary'
                  }`}>
                    {user?.full_name}
                  </span>
                )}
              </div>
            </Tooltip>
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-lg ${
                theme === 'dark'
                  ? 'hover:bg-dark-hover text-dark-text-secondary hover:text-dark-text-primary'
                  : 'hover:bg-light-hover text-light-text-secondary hover:text-light-text-primary'
              } transition-all duration-300 hover:scale-105`}
            >
              {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center justify-center w-full px-4 py-2 text-red-400 hover:bg-red-500/10 rounded-lg transition-all duration-300 transform hover:scale-105"
          >
            {isSidebarOpen ? (
              <>
                <LogOut className="w-5 h-5 mr-2" />
                <span>Logout</span>
              </>
            ) : (
              <LogOut className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        <div className={`${
          theme === 'dark' 
            ? 'bg-dark-sidebar border-dark-border' 
            : 'bg-white border-light-border'
        } p-4 border-b flex items-center justify-between shadow-md`}>
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className={`p-2 rounded-lg ${
              theme === 'dark'
                ? 'hover:bg-dark-hover text-primary'
                : 'hover:bg-light-hover text-primary'
            } transition-all duration-300 hover:scale-105`}
          >
            {isSidebarOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
          </button>

          <div className="flex items-center space-x-3">
            <Bot className="w-8 h-8 text-primary animate-pulse-subtle" />
            <div>
              <h2 className={`text-xl font-semibold ${
                theme === 'dark' ? 'text-dark-text-primary' : 'text-light-text-primary'
              }`}>
                RCA/PQA Supply Chain Analysis
              </h2>
              <p className={`text-sm ${
                theme === 'dark' ? 'text-dark-text-secondary' : 'text-light-text-secondary'
              }`}>
                AI-powered Root Cause & Predictive Quality Analysis
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <button className={`flex items-center space-x-2 ${
              theme === 'dark' 
                ? 'text-dark-text-secondary hover:text-dark-text-primary' 
                : 'text-light-text-secondary hover:text-light-text-primary'
              } transition-all duration-300 hover:scale-105`}>
              <Download className="w-5 h-5" />
              <span>Export</span>
            </button>
          </div>
        </div>

        <div className={`flex-1 overflow-y-auto p-4 space-y-4 ${
          theme === 'dark' ? 'bg-dark-bg' : 'bg-light-bg'
        }`}>
          {currentSession?.messages.map((message: Message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[80%] rounded-lg p-4 shadow-message hover:shadow-message-hover ${
                message.sender === 'user'
                  ? 'bg-primary text-secondary-dark'
                  : theme === 'dark'
                    ? 'bg-dark-sidebar text-dark-text-primary'
                    : 'bg-white text-light-text-primary border border-light-border'
              } transform transition-all duration-300 hover:scale-[1.02]`}>
                <p className="whitespace-pre-wrap">{message.text}</p>
                <span className={`text-xs ${
                  message.sender === 'user'
                    ? 'opacity-75'
                    : theme === 'dark'
                      ? 'text-dark-text-secondary'
                      : 'text-light-text-secondary'
                } block mt-2`}>
                  {new Date(message.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className={`p-4 border-t ${
          theme === 'dark' 
            ? 'border-dark-border bg-dark-sidebar' 
            : 'border-light-border bg-white shadow-lg'
        }`}>
          <div className="flex items-center space-x-2">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Type your supply chain query..."
              className={`flex-1 rounded-lg px-4 py-3 ${
                theme === 'dark'
                  ? 'bg-dark-hover text-dark-text-primary placeholder-dark-text-secondary'
                  : 'bg-light-hover text-light-text-primary placeholder-light-text-secondary border border-light-border'
              } focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-300 shadow-inner`}
              disabled={isLoading}
            />
            <button
              onClick={handleSendMessage}
              disabled={isLoading || !inputMessage.trim()}
              className="bg-primary text-secondary-dark p-3 rounded-lg hover:bg-primary-dark transition-all duration-300 transform hover:scale-105 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-6 h-6" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}