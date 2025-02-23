import { ChatProvider } from './context/ChatContext';
import { ChatDisplay } from './components/ChatDisplay';
import { AudioInput } from './components/AudioInput';
import './styles/ChatDisplay.css';

export const App: React.FC = () => {
  return (
    <ChatProvider>
      <div className="app-container">
        <ChatDisplay apiUrl={process.env.REACT_APP_API_URL || ''} />
        <AudioInput />
      </div>
    </ChatProvider>
  );
}; 