import axios, { AxiosInstance } from 'axios';

interface ProcessTextRequest {
  text: string;
  domain?: string;
  confidenceThreshold?: number;
}

interface ProcessImageRequest {
  image: File;
  confidenceThreshold?: number;
}

export class ModelAPI {
  private client: AxiosInstance;

  constructor(baseURL: string) {
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  async processText(request: ProcessTextRequest) {
    const payload = {
      modality: 'text',
      content: request.text,
      domain: request.domain,
      confidence_threshold: request.confidenceThreshold || 0.5,
    };

    return this.client.post('/process', payload);
  }

  async processImage(request: ProcessImageRequest) {
    const reader = new FileReader();
    
    return new Promise((resolve, reject) => {
      reader.onload = async () => {
        const base64Image = reader.result as string;
        
        const payload = {
          modality: 'image',
          content: base64Image.split(',')[1], // Remove data URL prefix
          confidence_threshold: request.confidenceThreshold || 0.5,
        };

        try {
          const response = await this.client.post('/process', payload);
          resolve(response);
        } catch (error) {
          reject(error);
        }
      };

      reader.onerror = () => reject(new Error('Failed to read image file'));
      reader.readAsDataURL(request.image);
    });
  }
} 