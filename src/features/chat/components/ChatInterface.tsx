import React from 'react';
import { useChat } from '../hooks/useChat';
import MessageList from './MessageList';
import ChatInput from './ChatInput';

export default function ChatInterface() {
  const { messages, isLoading, sendMessage } = useChat();

  return (
    <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-lg overflow-hidden h-[80vh] flex flex-col">
      <div className="flex-1 overflow-y-auto p-4">
        <MessageList messages={messages} />
      </div>
      <ChatInput onSend={sendMessage} isLoading={isLoading} />
    </div>
  );
}