# Add these diagnostic prints at the start of your file, after imports
print("PyTorch CUDA Diagnostics:")
print(f"CUDA is available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
print("GPU devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name()}")

import subprocess
import sys
import decord
import requests
import io
from PIL import Image
from transformers import Pix2StructForConditionalGeneration
from config.paths import ModelPaths, MODEL_CACHE
import os
from utils.banking_utils import BankingQueryParser

# Configure decord to use GPU if available
if torch.cuda.is_available():
    decord.bridge.set_bridge('torch')
    ctx = decord.gpu(0)
else:
    ctx = decord.cpu(0)

def install_dependencies():
    print("Installing required dependencies...")
    dependencies = [
        'vector-quantize-pytorch',
        'vocos',
        'transformers'
    ]
    for package in dependencies:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("All dependencies installed successfully!")

# Add this before model initialization
install_dependencies()

# First check CUDA availability
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))

# Model Configuration Section
class ModelConfig:
    MINICPM = {
        'name': 'openbmb/MiniCPM-2-6',
        'trust_remote_code': True,
        'attn_implementation': 'sdpa',
        'torch_dtype': torch.bfloat16,
        'init_vision': True,
        'init_audio': True,
        'init_tts': True
    }
    
    # Add new model configuration here
    NEW_MODEL = {
        'name': 'path/to/new/model',
        # Add specific configuration parameters
    }

    BANKING_LLM = {
        'name': 'unsloth/Meta-Llama-3.1-8B',
        'type': 'banking',
        'device_requirements': {
            'min_vram': 8,
            'dtype': None,  # Will be handled by FastLanguageModel
            'load_in_4bit': True,
            'max_seq_length': 2048
        },
        'model_class': 'FastLanguageModel',
        'quantization': '4bit'
    }

    ASR = {
        'name': 'Intel/whisper-large-int8-static-inc',
        'type': 'asr',
        'device_requirements': {
            'min_vram': 8,
            'dtype': None,  # Will be handled by ASR model
            'load_in_4bit': True,
            'max_seq_length': 2048
        },
        'model_class': 'WhisperForConditionalGeneration',
        'quantization': '4bit'
    }

    TTS = {
        'name': 'intronhealth/afro-tts',
        'type': 'tts',
        'device_requirements': {
            'min_vram': 4,
            'dtype': torch.float32
        },
        'model_class': 'AfroTTSModel',
        'processor': 'AfroTTSProcessor'
    }

    TRANSLATION = {
        'name': 'facebook/Seamless-M4T',
        'type': 'translation',
        'device_requirements': {
            'min_vram': 6,
            'dtype': torch.float16
        },
        'model_class': 'SeamlessM4TModel',
        'processor': 'SeamlessM4TProcessor'
    }

    DOLPHIN = {
        'name': 'cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF',
        'type': 'llm',
        'device_requirements': {
            'min_vram': 8,
            'dtype': torch.bfloat16,
            'load_in_8bit': True,
            'max_seq_length': 4096
        },
        'trust_remote_code': True
    }

class ModelLoader:
    def __init__(self, device):
        self.device = device
        self.models = {}
        self.tokenizers = {}
    
    def load_model(self, model_name, config):
        """Load a specific model and its tokenizer"""
        try:
            model = AutoModel.from_pretrained(**config)
            model = model.eval().to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(config['name'], trust_remote_code=True)
            
            if hasattr(model, 'init_tts'):
                model.init_tts()
                model.tts.float()
                
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            print(f"Successfully loaded {model_name}")
            
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
            
    def get_model(self, model_name):
        return self.models.get(model_name), self.tokenizers.get(model_name)

    def load_banking_model(self, specs):
        """Load banking-specific model"""
        try:
            print(f"Loading banking model: {specs['name']}")
            
            model_path = self._get_model_path(specs['name'])
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=specs['name'],
                max_seq_length=specs['device_requirements']['max_seq_length'],
                dtype=specs['device_requirements']['dtype'],
                load_in_4bit=specs['device_requirements']['load_in_4bit'],
                device_map="auto",
                cache_dir=model_path
            )
            
            self.models[specs['name']] = model
            self.tokenizers[specs['name']] = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Failed to load banking model: {str(e)}")
            return None, None

    def load_asr_model(self, specs):
        """Load ASR-specific model"""
        try:
            print(f"Loading ASR model: {specs['name']}")
            
            model_path = self._get_model_path(specs['name'])
            
            # Load model and processor
            model = WhisperForConditionalGeneration.from_pretrained(
                specs['name'],
                cache_dir=model_path,
                torch_dtype=specs['device_requirements']['dtype']
            ).to(self.device)
            
            processor = WhisperProcessor.from_pretrained(
                specs['name'],
                cache_dir=model_path
            )
            
            self.models[specs['name']] = model
            self.tokenizers[specs['name']] = processor
            
            return model, processor
            
        except Exception as e:
            print(f"Failed to load ASR model: {str(e)}")
            return None, None

    def load_tts_model(self, specs):
        """Load TTS-specific model"""
        try:
            print(f"Loading TTS model: {specs['name']}")
            
            model_path = self._get_model_path(specs['name'])
            
            # Load model and processor
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                specs['name'],
                cache_dir=model_path,
                torch_dtype=specs['device_requirements']['dtype']
            ).to(self.device)
            
            processor = AutoProcessor.from_pretrained(
                specs['name'],
                cache_dir=model_path
            )
            
            self.models[specs['name']] = model
            self.tokenizers[specs['name']] = processor
            
            return model, processor
            
        except Exception as e:
            print(f"Failed to load TTS model: {str(e)}")
            return None, None

    def load_translation_model(self, specs):
        """Load translation-specific model"""
        try:
            self.logger.info(f"Loading translation model: {specs['name']}")
            
            model_path = self._get_model_path(specs['name'])
            
            # Load model and processor
            model = SeamlessM4TModel.from_pretrained(
                specs['name'],
                cache_dir=model_path,
                torch_dtype=specs['device_requirements']['dtype']
            ).to(self.device)
            
            processor = SeamlessM4TProcessor.from_pretrained(
                specs['name'],
                cache_dir=model_path
            )
            
            self.models[specs['name']] = model
            self.tokenizers[specs['name']] = processor
            
            return model, processor
            
        except Exception as e:
            self.logger.error(f"Failed to load translation model: {str(e)}")
            return None, None

    def load_dolphin_model(self, specs):
        """Load Dolphin model"""
        try:
            self.logger.info(f"Loading Dolphin model: {specs['name']}")
            
            # Configure model loading
            model_config = {
                "device_map": "auto",
                "torch_dtype": specs['device_requirements']['dtype'],
                "load_in_8bit": True,
                "trust_remote_code": specs['trust_remote_code']
            }
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                specs['name'],
                **model_config
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                specs['name'],
                trust_remote_code=True
            )
            
            # Register the model
            self.models[specs['name']] = model
            self.tokenizers[specs['name']] = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load Dolphin model: {str(e)}")
            return None, None

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_loader = ModelLoader(device)

# Load models
model_loader.load_model('minicpm', ModelConfig.MINICPM)
# model_loader.load_model('new_model', ModelConfig.NEW_MODEL)  # Add when ready

# Get specific model for use
model, tokenizer = model_loader.get_model('minicpm')

# Initialize TTS
if hasattr(model, 'init_tts'):
    model.init_tts()
    model.tts.float()

# Initialize banking model
banking_model, banking_tokenizer = model_loader.load_banking_model(ModelConfig.BANKING_LLM)

# Initialize ASR model
asr_model, asr_processor = model_loader.load_asr_model(ModelConfig.ASR)

# Initialize TTS model
tts_model, tts_processor = model_loader.load_tts_model(ModelConfig.TTS)

# Initialize translation model
translation_model, translation_processor = model_loader.load_translation_model(ModelConfig.TRANSLATION)

# Initialize Dolphin model
dolphin_model, dolphin_tokenizer = model_loader.load_dolphin_model(ModelConfig.DOLPHIN)

import math
import numpy as np
from PIL import Image
from decord import VideoReader, cpu  # Changed from moviepy to decord
import tempfile
import librosa
import soundfile as sf

MAX_NUM_FRAMES = 64  # Fixed spacing

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    
    # Use the configured context
    vr = VideoReader(video_path, ctx=ctx)
    sample_fps = round(vr.get_avg_fps() / 1)  # Fixed method name
    frame_idx = [i for i in range(0, len(vr), sample_fps)]  # Fixed list comprehension
    if len(frame_idx) > MAX_NUM_FRAMES:  # Fixed comparison operator
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

video_path = '/root/joromigpt/fahdvideo.mp4'  # Updated path for droplet
frames = encode_video(video_path)
question = "Describe the Video"
msgs = [
    {'role': 'user', 'content': [question, *frames]}  # Fixed message format
]

params = {  # Fixed dictionary initialization
    "use_image_id": False,
    "max_slice_nums": 2
}

answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    **params
)
print(answer)

# Audio mimicking code
mimick_prompt = "Please repeat each user's speech, including voice style and speech content."
audio_path = '/root/joromigpt/common_voice_ig_41554715.mp3'  # Updated path
audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
msgs = [{'role': 'user', 'content': [mimick_prompt, audio_input]}]  # Fixed syntax

res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,  # Added missing comma
    max_new_tokens=128,
    use_tts_template=True,
    temperature=0.3,
    generate_audio=True,
    output_audio_path='/root/joromigpt/output.wav'  # Updated path
)

print(res)

class ModelTasks:
    @staticmethod
    def process_video(model, tokenizer, video_path, question):
        frames = encode_video(video_path)
        msgs = [{'role': 'user', 'content': [question, *frames]}]
        params = {"use_image_id": False, "max_slice_nums": 2}
        return model.chat(msgs=msgs, tokenizer=tokenizer, **params)
    
    @staticmethod
    def process_audio(model, tokenizer, audio_path, prompt):
        audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
        msgs = [{'role': 'user', 'content': [prompt, audio_input]}]
        return model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            max_new_tokens=128,
            use_tts_template=True,
            temperature=0.3,
            generate_audio=True,
            output_audio_path='/root/joromigpt/output.wav'
        )

    @staticmethod
    def process_banking(model, tokenizer, query):
        # Format input using Alpaca template
        input_text = f'''Below is an instruction that describes a task, paired with an appropriate response.

## Instruction:
User Query: {query}

### Input:
None

### Response:
Answer:'''
        
        # Prepare inputs
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.95
        )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer and explanation
        parts = response.split('Answer:')
        if len(parts) > 1:
            answer = parts[1].strip()
        else:
            answer = response
            
        return {
            "status": "success",
            "text": answer,
            "model_type": "banking"
        }

    @staticmethod
    def process_speech(model, processor, audio_input):
        """Process speech input to text"""
        try:
            # Process the audio
            input_features = processor(
                audio_input, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(model.device)
            
            # Generate transcription
            predicted_ids = model.generate(input_features)
            
            # Decode the output
            transcription = processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return {
                "status": "success",
                "text": transcription,
                "model_type": "asr"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_type": "general",
                "message": f"ASR processing failed: {str(e)}"
            }

    @staticmethod
    def generate_speech(model, processor, text, accent):
        """Generate speech from text with African accents"""
        try:
            # Prepare inputs with accent conditioning
            inputs = processor(
                text=text,
                accent=accent,  # Will be None if no specific accent requested
                return_tensors="pt"
            ).to(model.device)
            
            # Generate speech
            speech_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7
            )
            
            # Convert to audio
            speech = processor.batch_decode(speech_ids)[0]
            
            # Get available accents
            available_accents = processor.available_accents if hasattr(processor, 'available_accents') else []
            
            return {
                "status": "success",
                "audio": speech.cpu().numpy(),
                "sampling_rate": processor.sampling_rate,
                "accent_used": accent,
                "available_accents": available_accents
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_type": "general",
                "message": f"Speech generation failed: {str(e)}"
            }

    @staticmethod
    def translate(model, processor, text, source_lang, target_lang):
        """Translate text between languages"""
        try:
            self.logger.info(f"Translating from {source_lang} to {target_lang}")
            
            # Prepare inputs
            inputs = processor(
                text=text,
                src_lang=source_lang,
                tgt_lang=target_lang,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate translation
            output_tokens = model.generate(
                **inputs,
                tgt_lang=target_lang,
                max_length=256
            )
            
            # Decode translation
            translation = processor.decode(
                output_tokens[0].tolist(),
                skip_special_tokens=True
            )
            
            return {
                "status": "success",
                "text": translation,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "model_type": "translation"
            }
            
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            raise

    @staticmethod
    def generate_text(model, tokenizer, prompt, max_length, temperature):
        """Generate text using Dolphin model"""
        try:
            # Prepare inputs
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(model.device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode output
            generated_text = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            return {
                "status": "success",
                "generated_text": generated_text,
                "model_type": "dolphin"
            }
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "model_type": "dolphin"
            }

def test_visual_qa():
    # API endpoint
    url = "http://localhost:8000/image"
    
    # Load test image
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Prepare request
    files = {
        'file': ('image.png', img_byte_arr, 'image/png')
    }
    
    data = {
        'question': "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud",
        'language': 'en',
        'output_modality': 'text'
    }
    
    # Send request
    response = requests.post(url, files=files, data=data)
    print(response.json())

def test_banking_queries():
    test_cases = [
        "How do I block my credit card?",
        "What's my account balance?",
        "I noticed suspicious activity on my account",
        "How do I make a transfer?",
    ]
    
    for query in test_cases:
        print(f"\nTesting query: {query}")
        query_type = BankingQueryParser.classify_query(query)
        print(f"Query type: {query_type.value}")
        
        response = ModelTasks.process_banking(
            banking_model,
            banking_tokenizer,
            query
        )
        print(f"Response: {response['text']}")
        if response.get('requires_action'):
            print("⚠️ This query requires additional action!")

def test_speech_recognition():
    # Load audio file
    audio_path = '/path/to/audio/file.wav'
    audio_input, _ = librosa.load(audio_path, sr=16000)
    
    # Process speech
    response = ModelTasks.process_speech(
        asr_model,
        asr_processor,
        audio_input
    )
    print(f"Transcription: {response['text']}")

def test_tts_enhanced():
    text = "Welcome to our banking service. How may I assist you today?"
    
    # Test different combinations
    test_cases = [
        {"accent": "nigerian", "style": "formal", "rate": 1.0, "pitch_shift": 0.0},
        {"accent": "kenyan", "style": "casual", "rate": 1.2, "pitch_shift": 0.2},
        {"accent": "south_african", "style": "elderly", "rate": 0.9, "pitch_shift": -0.3}
    ]
    
    for case in test_cases:
        response = ModelTasks.generate_speech(
            tts_model,
            tts_processor,
            text,
            **case
        )
        
        if response["status"] == "success":
            output_path = f"output_{case['accent']}_{case['style']}.wav"
            sf.write(
                output_path,
                response["audio"],
                response["sampling_rate"]
            )
            print(f"Generated speech saved to {output_path}")
            print(f"Parameters used: {case}")
            print(f"Available accents: {response['available_accents']}")
            print(f"Available styles: {response['available_styles']}")

def test_translation_enhanced():
    # Test language detection and translation
    test_cases = [
        # Single text with auto-detection
        {
            "text": "Comment allez-vous?",
            "target_lang": "eng",
            "detect_language": True
        },
        # Batch translation
        {
            "text": [
                "How are you?",
                "¿Cómo estás?",
                "Comment allez-vous?"
            ],
            "target_lang": "swa",  # Translate to Swahili
            "detect_language": True
        }
    ]
    
    for case in test_cases:
        response = ModelTasks.translate(
            translation_model,
            translation_processor,
            case["text"],
            target_lang=case["target_lang"],
            detect_language=case.get("detect_language", False)
        )
        
        if response["status"] == "success":
            if isinstance(case["text"], list):
                print("\nBatch Translation Results:")
                for i, trans in enumerate(response["translations"]):
                    print(f"\nText {i+1}:")
                    print(f"Original: {trans['original']}")
                    print(f"Detected: {response['detected_languages'][i][0]}")
                    print(f"Translated: {trans['translated']}")
            else:
                print(f"\nSingle Translation:")
                print(f"Original: {case['text']}")
                print(f"Translated: {response['text']}")
                print(f"Source Language: {response['source_lang']}")

def test_dolphin_generation():
    """Test Dolphin model text generation"""
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short story about a time traveler.",
        "What are the main challenges in machine learning?"
    ]
    
    for prompt in test_prompts:
        response = ModelTasks.generate_text(
            dolphin_model,
            dolphin_tokenizer,
            prompt,
            max_length=512,
            temperature=0.7
        )
        
        if response["status"] == "success":
            print(f"\nPrompt: {prompt}")
            print(f"Generated Text: {response['generated_text']}")
        else:
            print(f"Error: {response.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_banking_queries()
    test_speech_recognition()
    test_tts_enhanced()
    test_translation_enhanced()
    test_dolphin_generation()

# Set cache directory before importing transformers
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE)

# Now load Pix2Struct
model = Pix2StructForConditionalGeneration.from_pretrained(
    "google/pix2struct-ai2d-base",
    cache_dir=ModelPaths.PIX2STRUCT
)
print(f"Model loaded from: {model.config._name_or_path}") 