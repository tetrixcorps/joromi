import React, { useEffect, useRef, useState } from 'react';
import { useChat } from '../context/ChatContext';
import { useWebSocket } from '../hooks/useWebSocket';
import { useSpeechSynthesis } from '../hooks/useSpeechSynthesis';
import { LanguageSelector } from './LanguageSelector';

interface ChatDisplayProps {
  apiUrl: string;
}

export const ChatDisplay: React.FC<ChatDisplayProps> = ({ apiUrl }) => {
  const { state, dispatch } = useChat();
  const chatEndRef = useRef<HTMLDivElement>(null);
  const ws = useWebSocket(`${apiUrl}/ws/chat`);
  const { speak, speaking } = useSpeechSynthesis();
  const [selectedLanguage, setSelectedLanguage] = useState("eng");

  useEffect(() => {
    if (ws) {
      ws.onmessage = (event) => {
        const response = JSON.parse(event.data);
        
        if (response.type === 'transcription') {
          // Handle live transcription updates
          dispatch({
            type: 'ADD_MESSAGE',
            payload: {
              id: Date.now().toString(),
              text: response.text,
              type: 'user',
              timestamp: new Date(),
            },
          });
        } else if (response.type === 'response') {
          // Handle model responses
          const message = {
            id: Date.now().toString(),
            text: response.text,
            type: 'assistant',
            timestamp: new Date(),
          };
          
          dispatch({ type: 'ADD_MESSAGE', payload: message });
          
          // Synthesize speech if enabled
          if (response.synthesize_speech) {
            speak(response.text);
          }
        }
      };
    }
  }, [ws, dispatch, speak]);

  // Scroll to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [state.messages]);

  const handleLanguageChange = (language: string) => {
    setSelectedLanguage(language);
    ws.sendMessage({
      type: "set_language",
      language: language
    });
  };

  return (
    <div className="chat-display">
      <LanguageSelector
        onLanguageChange={handleLanguageChange}
        currentLanguage={selectedLanguage}
      />
      <div className="messages-container">
        {state.messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.type}`}
          >
            <div className="message-content">
              <p>{message.text}</p>
              {message.type === 'assistant' && !speaking && (
                <button
                  onClick={() => speak(message.text)}
                  className="play-audio"
                >
                  🔊
                </button>
              )}
            </div>
            <div className="message-timestamp">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>
      
      {state.isProcessing && (
        <div className="processing-indicator">
          Processing...
        </div>
      )}
    </div>
  );
}; 