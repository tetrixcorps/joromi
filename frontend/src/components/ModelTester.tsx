import React, { useState } from 'react';
import { ModelAPI } from '../services/model-api';

const api = new ModelAPI(process.env.REACT_APP_API_URL || 'http://localhost:8000');

export const ModelTester: React.FC = () => {
  const [result, setResult] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTextSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setError(null);

    const formData = new FormData(event.currentTarget);
    
    try {
      const response = await api.processText({
        text: formData.get('text') as string,
        domain: formData.get('domain') as string,
        confidenceThreshold: Number(formData.get('confidence'))
      });
      
      setResult(JSON.stringify(response.data, null, 2));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="model-tester">
      <form onSubmit={handleTextSubmit}>
        <div>
          <label htmlFor="text">Text Input:</label>
          <textarea
            id="text"
            name="text"
            rows={4}
            required
          />
        </div>

        <div>
          <label htmlFor="domain">Domain:</label>
          <select id="domain" name="domain">
            <option value="general">General</option>
            <option value="medical">Medical</option>
            <option value="technical">Technical</option>
          </select>
        </div>

        <div>
          <label htmlFor="confidence">Confidence Threshold:</label>
          <input
            type="number"
            id="confidence"
            name="confidence"
            step="0.1"
            min="0"
            max="1"
            defaultValue="0.5"
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Submit'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}
      {result && (
        <pre className="result">
          {result}
        </pre>
      )}
    </div>
  );
}; 