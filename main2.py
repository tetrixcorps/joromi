import unittest
import torch
import os
from decord import VideoReader
from PIL import Image
import librosa

class TestDropletSetup(unittest.TestCase):
    def test_cuda_setup(self):
        """Test CUDA availability and setup"""
        print("\nCUDA Setup Information:")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            
            # Test CUDA computation
            x = torch.rand(5, 5).cuda()
            y = torch.rand(5, 5).cuda()
            z = x @ y
            self.assertTrue(z.is_cuda, "CUDA computation failed")
        else:
            self.fail("CUDA is not available")

    def test_paths(self):
        """Test if files and directories exist"""
        base_dir = '/root/joromigpt'
        paths = {
            'video': f'{base_dir}/fahdvideo.mp4',
            'audio': f'{base_dir}/common_voice_ig_41554715.mp3',
            'output': base_dir  # Directory for output files
        }
        
        # Test directory
        self.assertTrue(os.path.isdir(base_dir), f"Base directory not found: {base_dir}")
        
        # Test input files
        for name, path in paths.items():
            if name != 'output':
                self.assertTrue(os.path.isfile(path), f"File not found: {path}")
            
        # Test write permission
        test_file = os.path.join(base_dir, 'test_write.tmp')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            self.fail(f"Directory not writable: {str(e)}")

    def test_video_loading(self):
        """Test video loading and frame extraction"""
        video_path = '/root/joromigpt/fahdvideo.mp4'
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            self.assertGreater(len(vr), 0, "Video has no frames")
            
            # Test frame extraction
            frame = vr[0].asnumpy()
            img = Image.fromarray(frame.astype('uint8'))
            self.assertIsInstance(img, Image.Image, "Failed to convert frame to PIL Image")
        except Exception as e:
            self.fail(f"Video loading failed: {str(e)}")

    def test_audio_loading(self):
        """Test audio loading"""
        audio_path = '/root/joromigpt/common_voice_ig_41554715.mp3'
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            self.assertEqual(sr, 16000, "Incorrect sample rate")
            self.assertGreater(len(audio), 0, "Audio is empty")
        except Exception as e:
            self.fail(f"Audio loading failed: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2) 