from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from config.training_config import FineTuneConfig
from models.model_loader import ModelManager
from config.model_config import ModelConfigurations

def setup_fine_tuning():
    # Initialize model manager
    model_manager = ModelManager(device='cuda')
    
    # Load base model
    model, processor = model_manager.load_translation_model(
        ModelConfigurations.TRANSLATION
    )
    
    # Prepare training arguments
    training_args = Seq2SeqTrainingArguments(
        **FineTuneConfig.SEAMLESS_M4T
    )
    
    return model, processor, training_args

def main():
    model, processor, training_args = setup_fine_tuning()
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
    )
    
    # Start fine-tuning
    trainer.train()
    
    # Save fine-tuned model
    trainer.save_model()

if __name__ == "__main__":
    main() 