import React from 'react';
import { Message } from '../types';
import { Bot, User } from 'lucide-react';

interface MessageItemProps {
  message: Message;
}

export default function MessageItem({ message }: MessageItemProps) {
  const isBot = message.role === 'assistant';

  return (
    <div className={`flex ${isBot ? 'justify-start' : 'justify-end'}`}>
      <div
        className={`flex items-start space-x-2 max-w-[70%] ${
          isBot ? 'flex-row' : 'flex-row-reverse'
        }`}
      >
        <div className={`p-2 rounded-full ${isBot ? 'bg-gray-100' : 'bg-indigo-100'}`}>
          {isBot ? <Bot size={20} /> : <User size={20} />}
        </div>
        
        <div
          className={`rounded-lg p-3 ${
            isBot
              ? 'bg-gray-100 text-gray-800'
              : 'bg-indigo-600 text-white'
          }`}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
          <span className="text-xs opacity-75 block mt-1">
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>
      </div>
    </div>
  );
}