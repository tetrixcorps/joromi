import os
import librosa
import unittest
import torch
import decord
from decord import VideoReader
from PIL import Image

# Configure decord to use GPU if available
if torch.cuda.is_available():
    decord.bridge.set_bridge('torch')
    ctx = decord.gpu(0)
else:
    ctx = decord.cpu(0)

class TestVideoLoading(unittest.TestCase):
    def test_video_loading(self):
        """Test video loading with automatic device selection"""
        video_path = '/root/joromigpt/fahdvideo.mp4'
        try:
            # Print device context being used
            print(f"\nUsing device context: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            
            # Load video with appropriate context
            vr = VideoReader(video_path, ctx=ctx)
            print(f"Video loaded successfully with {len(vr)} frames")
            
            # Test frame extraction
            frame = vr[0].asnumpy()
            img = Image.fromarray(frame.astype('uint8'))
            self.assertIsInstance(img, Image.Image, "Frame conversion failed")
            print("Frame extraction successful")
            
            return vr
        except Exception as e:
            self.fail(f"Video loading failed: {str(e)}")

class TestAudioLoading(unittest.TestCase):
    def test_audio_loading(self):
        """Test audio loading"""
        audio_path = '/root/joromigpt/common_voice_ig_41554715.mp3'
        
        # First check if file exists and is readable
        self.assertTrue(os.path.exists(audio_path), f"Audio file not found: {audio_path}")
        self.assertTrue(os.access(audio_path, os.R_OK), f"Audio file not readable: {audio_path}")
        
        try:
            audio, sr = librosa.load(
                audio_path, 
                sr=16000, 
                mono=True,
                duration=None
            )
            self.assertEqual(sr, 16000, "Incorrect sample rate")
            self.assertGreater(len(audio), 0, "Audio is empty")
            print(f"Successfully loaded audio: {len(audio)} samples at {sr}Hz")
        except Exception as e:
            self.fail(f"Audio loading failed: {str(e)}")

class TestDropletSetup(unittest.TestCase):
    def test_cuda_setup(self):
        """Test CUDA availability and setup"""
        print("\nCUDA Setup Information:")
        try:
            cuda_available = torch.cuda.is_available()
            print(f"CUDA available: {cuda_available}")
            
            if not cuda_available:
                print("\nRunning in CPU mode")
                print("PyTorch version:", torch.__version__)
                return
            
            # CUDA tests
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            
            # Test CUDA computation
            print("\nTesting CUDA computation...")
            x = torch.rand(5, 5).cuda()
            y = torch.rand(5, 5).cuda()
            z = x @ y
            self.assertTrue(z.is_cuda, "CUDA computation failed")
            print("CUDA computation successful")
            
            # Memory info
            print("\nGPU Memory Status:")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            
            torch.cuda.empty_cache()
            print("CUDA cache cleared")
            
        except Exception as e:
            self.fail(f"CUDA setup failed: {str(e)}")

def run_tests():
    """Run tests with device information"""
    print(f"Running tests on: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_tests() 