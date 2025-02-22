class FineTuneConfig:
    SEAMLESS_M4T = {
        'learning_rate': 2e-5,
        'batch_size': 8,
        'gradient_accumulation_steps': 4,
        'num_epochs': 3,
        'warmup_steps': 100,
        'save_steps': 500,
        'evaluation_strategy': 'steps',
        'eval_steps': 100,
        'logging_steps': 50,
        'output_dir': 'fine_tuned_seamless_m4t',
        'fp16': True,
        'gradient_checkpointing': True
    } 