import React, { useState, useRef } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

export const AudioInput: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const ws = useWebSocket('ws://your-api-gateway/audio');

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      
      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          ws.send(event.data);
        }
      };

      mediaRecorder.current.start(100); // Send chunks every 100ms
      setIsRecording(true);
    } catch (err) {
      console.error('Failed to start recording:', err);
    }
  };

  return (
    <div className="audio-input">
      <button 
        onClick={() => isRecording ? stopRecording() : startRecording()}
        className={isRecording ? 'recording' : ''}
      >
        {isRecording ? 'Stop Recording' : 'Start Recording'}
      </button>
    </div>
  );
}; 