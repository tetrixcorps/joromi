import { useEffect, useRef, useCallback } from 'react';
import { useChat } from '../context/ChatContext';

export const useWebSocket = (url: string) => {
  const ws = useRef<WebSocket | null>(null);
  const { dispatch } = useChat();

  const sendMessage = useCallback((data: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.current.onmessage = (event) => {
      const response = JSON.parse(event.data);
      
      if (response.type === 'transcription_start') {
        dispatch({ type: 'SET_PROCESSING', payload: true });
      } else if (response.type === 'transcription_end') {
        dispatch({ type: 'SET_PROCESSING', payload: false });
      }
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
    };

    return () => {
      ws.current?.close();
    };
  }, [url, dispatch]);

  return {
    sendMessage,
    isConnected: ws.current?.readyState === WebSocket.OPEN,
  };
}; 