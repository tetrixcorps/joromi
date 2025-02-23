import axios from 'axios';
import { ModelAPI } from '../src/services/model-api';

describe('Model API Integration Tests', () => {
  const api = new ModelAPI('http://localhost:8000');

  test('text processing with general model', async () => {
    const response = await api.processText({
      text: 'What is machine learning?',
      domain: 'general',
      confidenceThreshold: 0.5
    });

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('response');
  });

  test('image processing', async () => {
    const imageFile = new File([''], 'test-image.jpg');
    const response = await api.processImage({
      image: imageFile,
      confidenceThreshold: 0.7
    });

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('response');
  });

  test('domain-specific processing', async () => {
    const response = await api.processText({
      text: 'What are the symptoms of COVID-19?',
      domain: 'medical',
      confidenceThreshold: 0.9
    });

    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('response');
  });
}); 