#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import logging
from typing import List, Dict
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

class SetupVerifier:
    def __init__(self):
        self.project_root = Path("/home/kali/joromigpt/african_translation")
        self.required_files = {
            "Main Files": [
                "app.py",
                "run_all.py",
                "setup.py",
                "requirements.txt",
                "__init__.py"
            ],
            "Scripts": [
                "scripts/__init__.py",
                "scripts/finetune_translation.py",
                "scripts/setup_nltk.py",
                "scripts/verify_setup.py"
            ],
            "Utils": [
                "utils/__init__.py",
                "utils/african_language_utils.py"
            ]
        }
        
        self.required_dirs = [
            "scripts",
            "utils",
            "logs",
            "models"
        ]
        
        self.dataset_paths = {
            "Yoruba": "/home/kali/Desktop/Lang/yo/clips",
            "Hausa": "/home/kali/Desktop/Lang/ha/clips",
            "Igbo": "/home/kali/Desktop/Lang/ig/clips"
        }
        
        os.makedirs(self.project_root / "logs", exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.project_root / "logs/setup_verification.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def print_status(self, message: str, status: bool):
        """Print colored status message"""
        status_text = f"{Fore.GREEN}[✓]" if status else f"{Fore.RED}[✗]"
        print(f"{status_text} {message}{Style.RESET_ALL}")
    
    def verify_directory_structure(self) -> bool:
        """Verify all required directories exist"""
        print(f"\n{Fore.CYAN}Checking Directory Structure:{Style.RESET_ALL}")
        all_valid = True
        
        for dir_name in self.required_dirs:
            dir_path = self.project_root / dir_name
            exists = dir_path.exists() and dir_path.is_dir()
            self.print_status(f"Directory '{dir_name}'", exists)
            if not exists:
                all_valid = False
                self.logger.error(f"Missing directory: {dir_name}")
        
        return all_valid
    
    def verify_files(self) -> bool:
        """Verify all required files exist"""
        print(f"\n{Fore.CYAN}Checking Required Files:{Style.RESET_ALL}")
        all_valid = True
        
        for category, files in self.required_files.items():
            print(f"\n{Fore.YELLOW}{category}:{Style.RESET_ALL}")
            for file_path in files:
                full_path = self.project_root / file_path
                exists = full_path.exists() and full_path.is_file()
                self.print_status(f"File '{file_path}'", exists)
                if not exists:
                    all_valid = False
                    self.logger.error(f"Missing file: {file_path}")
        
        return all_valid
    
    def verify_dataset_paths(self) -> bool:
        """Verify dataset directories exist and contain required files"""
        print(f"\n{Fore.CYAN}Checking Dataset Paths:{Style.RESET_ALL}")
        all_valid = True
        
        for language, path in self.dataset_paths.items():
            path = Path(path)
            print(f"\n{Fore.YELLOW}{language} Dataset:{Style.RESET_ALL}")
            
            # Check directory exists
            dir_exists = path.exists() and path.is_dir()
            self.print_status(f"Directory '{path}'", dir_exists)
            
            if dir_exists:
                # Check for audio files
                audio_files = list(path.glob("*.wav"))
                has_audio = len(audio_files) > 0
                self.print_status(f"Audio files (.wav)", has_audio)
                
                # Check for transcription files
                trans_files = list(path.glob("*.txt"))
                has_trans = len(trans_files) > 0
                self.print_status(f"Transcription files (.txt)", has_trans)
                
                # Check for translation files
                eng_files = list(path.glob("*.eng"))
                has_eng = len(eng_files) > 0
                self.print_status(f"Translation files (.eng)", has_eng)
                
                if not (has_audio and has_trans and has_eng):
                    all_valid = False
                    self.logger.error(f"Missing required files in {language} dataset")
            else:
                all_valid = False
                self.logger.error(f"Dataset directory not found: {path}")
        
        return all_valid
    
    def verify_dependencies(self) -> bool:
        """Verify all required Python packages are installed"""
        print(f"\n{Fore.CYAN}Checking Python Dependencies:{Style.RESET_ALL}")
        all_valid = True
        
        required_packages = [
            "torch",
            "transformers",
            "datasets",
            "sacrebleu",
            "nltk",
            "numpy",
            "pandas"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                self.print_status(f"Package '{package}'", True)
            except ImportError:
                all_valid = False
                self.print_status(f"Package '{package}'", False)
                self.logger.error(f"Missing Python package: {package}")
        
        return all_valid
    
    def run_verification(self) -> bool:
        """Run all verification checks"""
        print(f"{Fore.BLUE}Starting Setup Verification...{Style.RESET_ALL}")
        
        checks = [
            self.verify_directory_structure(),
            self.verify_files(),
            self.verify_dataset_paths(),
            self.verify_dependencies()
        ]
        
        all_passed = all(checks)
        
        print(f"\n{Fore.BLUE}Verification Complete:{Style.RESET_ALL}")
        if all_passed:
            print(f"{Fore.GREEN}All checks passed successfully!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Some checks failed. Please check the log file for details.{Style.RESET_ALL}")
        
        return all_passed

def main():
    verifier = SetupVerifier()
    success = verifier.run_verification()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 