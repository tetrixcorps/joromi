import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from transformers import (
    SeamlessM4TModel,
    SeamlessM4TProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Dataset, Audio
import numpy as np
from typing import Dict, List
import logging
import pandas as pd
import os

# Now utils can be imported
from utils.african_language_utils import (
    AfricanLanguagePreprocessor,
    DataAugmenter,
    AfricanLanguageEvaluator,
    TranslationCache
)

class AfricanLanguageDatasetPreprocessor:
    def __init__(self, base_path: str, language: str):
        self.base_path = Path(base_path)
        self.language = language
        self.clips_path = self.base_path / language / "clips"
        
    def load_and_prepare_data(self) -> Dataset:
        """Load and prepare audio-text pairs from clips directory"""
        # Get all audio files and their transcriptions
        audio_files = []
        transcriptions = []
        english_translations = []
        
        for audio_file in self.clips_path.glob("*.wav"):
            # Get corresponding transcription file
            trans_file = audio_file.with_suffix('.txt')
            if trans_file.exists():
                with open(trans_file, 'r', encoding='utf-8') as f:
                    transcription = f.read().strip()
                    
                # Get English translation (assuming it's in a parallel file)
                eng_file = audio_file.with_suffix('.eng')
                if eng_file.exists():
                    with open(eng_file, 'r', encoding='utf-8') as f:
                        translation = f.read().strip()
                else:
                    continue  # Skip if no translation available
                
                audio_files.append(str(audio_file))
                transcriptions.append(transcription)
                english_translations.append(translation)
        
        # Create dataset
        dataset_dict = {
            "audio": audio_files,
            "transcription": transcriptions,
            "translation": english_translations
        }
        
        # Convert to Hugging Face dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Add audio feature
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        return dataset
    
    def split_dataset(self, dataset: Dataset, train_size: float = 0.8) -> Dict[str, Dataset]:
        """Split dataset into train and validation sets"""
        train_test = dataset.train_test_split(train_size=train_size)
        return {
            "train": train_test["train"],
            "validation": train_test["test"]
        }

class AfricanLanguageTranslationTrainer:
    def __init__(
        self,
        language: str,
        base_model: str = "facebook/seamless-m4t-v2",
        output_dir: str = None,
        device: str = "cuda"
    ):
        self.language = language
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = SeamlessM4TModel.from_pretrained(base_model).to(self.device)
        self.processor = SeamlessM4TProcessor.from_pretrained(base_model)
        self.output_dir = Path(output_dir or f"fine_tuned_{language}_translation")
        self.logger = logging.getLogger(__name__)
        self.preprocessor = AfricanLanguagePreprocessor(language)
        self.augmenter = DataAugmenter(language)
        self.evaluator = AfricanLanguageEvaluator(language)
        self.translation_cache = TranslationCache()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{language}_training.log"),
                logging.StreamHandler()
            ]
        )
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset with augmentation and preprocessing"""
        def preprocess_function(examples):
            # Normalize and preprocess text
            processed_translations = [
                self.preprocessor.normalize_text(text)
                for text in examples["translation"]
            ]
            processed_transcriptions = [
                self.preprocessor.normalize_text(text)
                for text in examples["transcription"]
            ]
            
            # Apply augmentation
            augmented_texts = []
            augmented_translations = []
            for text, trans in zip(processed_transcriptions, processed_translations):
                aug_texts = self.augmenter.augment_text(text)
                augmented_texts.extend(aug_texts)
                augmented_translations.extend([trans] * len(aug_texts))
            
            # Add augmented data
            processed_transcriptions.extend(augmented_texts)
            processed_translations.extend(augmented_translations)
            
            # Process source text (English)
            source_inputs = self.processor(
                text=processed_translations,
                src_lang="eng",
                tgt_lang=self.language,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Process target text (African language)
            target_inputs = self.processor(
                text=processed_transcriptions,
                src_lang=self.language,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            return {
                "input_ids": source_inputs.input_ids,
                "attention_mask": source_inputs.attention_mask,
                "labels": target_inputs.input_ids
            }
        
        return dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate model with custom metrics"""
        predictions = []
        references = []
        
        for batch in eval_dataset:
            # Generate prediction
            outputs = self.model.generate(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device)
            )
            
            # Decode prediction
            pred_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            ref_text = self.processor.batch_decode(
                batch["labels"],
                skip_special_tokens=True
            )
            
            predictions.extend(pred_text)
            references.extend(ref_text)
        
        # Evaluate using custom metrics
        metrics = self.evaluator.evaluate(
            predictions,
            references,
            metrics=['bleu', 'tone_accuracy', 'dialect_consistency']
        )
        
        return metrics
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        training_args: Dict = None
    ):
        """Fine-tune the model"""
        if training_args is None:
            training_args = Seq2SeqTrainingArguments(
                output_dir=str(self.output_dir),
                evaluation_strategy="steps",
                eval_steps=100,
                learning_rate=2e-5,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                gradient_accumulation_steps=4,
                weight_decay=0.01,
                save_total_limit=3,
                num_train_epochs=5,
                predict_with_generate=True,
                fp16=True,
                push_to_hub=False,
                logging_dir=f"logs/{self.language}",
                logging_steps=50
            )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor,
            data_collator=DataCollatorForSeq2Seq(
                self.processor,
                model=self.model,
                padding=True
            )
        )
        
        # Train
        self.logger.info(f"Starting training for {self.language}...")
        trainer.train()
        
        # Save fine-tuned model
        trainer.save_model()
        self.processor.save_pretrained(self.output_dir)
        
        return trainer

def main():
    # Base path for all language datasets
    base_path = "/home/kali/Desktop/Lang"
    
    # Languages to process
    languages = {
        "yo": "yor",  # Yoruba
        "ha": "hau",  # Hausa
        "ig": "ibo"   # Igbo
    }
    
    for lang_code, lang_id in languages.items():
        # Initialize preprocessor
        preprocessor = AfricanLanguageDatasetPreprocessor(base_path, lang_code)
        
        # Load and prepare dataset
        dataset = preprocessor.load_and_prepare_data()
        splits = preprocessor.split_dataset(dataset)
        
        # Initialize trainer
        trainer = AfricanLanguageTranslationTrainer(
            language=lang_id,
            output_dir=f"fine_tuned_{lang_id}_translation"
        )
        
        # Prepare datasets for training
        train_dataset = trainer.prepare_dataset(splits["train"])
        eval_dataset = trainer.prepare_dataset(splits["validation"])
        
        # Train
        trainer.train(train_dataset, eval_dataset)
        
        # Log completion
        print(f"Completed training for {lang_id}")

if __name__ == "__main__":
    main() 