#!/usr/bin/env python3
"""
Demo script showing how to add TrainerControlCallback to your training job.

This script demonstrates:
1. How to add the TrainerControlCallback to your trainer
2. How to discover and control the running job from another terminal
3. Best practices for using the control system

Usage:
  # Terminal 1: Start training with control enabled
  python examples/trainer_control_demo.py
  
  # Terminal 2: Control the training job
  forgather control list                    # Find your job
  forgather control status JOB_ID          # Check status  
  forgather control save JOB_ID            # Save checkpoint
  forgather control stop JOB_ID            # Gracefully stop
"""

import sys
import os
import time
import platform

# Add forgather to path if needed (when running from source)
try:
    import forgather
except ImportError:
    # Try to find forgather source directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    forgather_src = os.path.join(current_dir, '..', '..', 'src')
    if os.path.exists(forgather_src):
        sys.path.insert(0, forgather_src)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from forgather.ml.trainer.trainer import Trainer
from forgather.ml.trainer.trainer_types import TrainingArguments
from forgather.ml.trainer.callbacks import TrainerControlCallback, JsonLogger


class DemoDataset(Dataset):
    """Demo dataset for training."""
    
    def __init__(self, size=1000, seq_len=32, vocab_size=1000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random sequences for language modeling
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = torch.roll(input_ids, -1)
        labels[-1] = torch.randint(0, self.vocab_size, (1,))
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class DemoModel(nn.Module):
    """Simple transformer-like model for demo."""
    
    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, labels=None):
        # Embedding
        x = self.embedding(input_ids)
        
        # Transformer
        x = self.transformer(x)
        
        # Output projection
        logits = self.output_proj(x)
        
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            return (loss, logits)
        
        return (logits,)


def create_job_id():
    """Create a descriptive job ID."""
    timestamp = int(time.time())
    hostname = platform.node()
    return f"demo_training_{hostname}_{timestamp}"


def main():
    """Main training function with control callback enabled."""
    
    print("🚀 Starting Forgather trainer control demo")
    print("="*60)
    
    # Create job ID
    job_id = create_job_id()
    print(f"Job ID: {job_id}")
    
    # Model and data
    print("📝 Setting up model and data...")
    model = DemoModel(vocab_size=1000, hidden_size=256, num_layers=2)
    train_dataset = DemoDataset(size=2000, seq_len=32)
    eval_dataset = DemoDataset(size=500, seq_len=32)
    
    # Training arguments
    args = TrainingArguments(
        output_dir=f"./demo_output/{job_id}",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        max_steps=500,  # Enough steps to test control
        
        # Logging and evaluation
        logging_steps=10,
        eval_steps=50, 
        eval_strategy="steps",
        
        # Checkpointing
        save_steps=100,
        save_strategy="steps", 
        save_total_limit=3,
        
        # Optimizer settings
        learning_rate=1e-4,
        weight_decay=0.01,
        
        # Use CPU if no free GPU
        use_cpu=not torch.cuda.is_available() or torch.cuda.current_device() not in [2, 3],
    )
    
    # Create callbacks
    callbacks = [
        # Enable trainer control with custom job ID
        TrainerControlCallback(
            job_id=job_id,
            # port=None,  # Auto-select port
            # enable_http=True  # Enable HTTP control (default)
        ),
        
        # JSON logging for better monitoring
        JsonLogger(),
    ]
    
    # Create trainer
    print("🏋️  Creating trainer...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks
    )
    
    print("✅ Trainer created successfully!")
    print()
    print("🎛️  Control Interface Information:")
    print(f"   Job ID: {job_id}")
    print(f"   Host: {platform.node()}")
    print(f"   Discovery: ~/.forgather/jobs/{job_id}/endpoint.json")
    print()
    print("📱 Control Commands (run in another terminal):")
    print(f"   forgather control list                    # List all jobs")
    print(f"   forgather control status {job_id}       # Get job status")
    print(f"   forgather control save {job_id}         # Save checkpoint")
    print(f"   forgather control stop {job_id}         # Graceful stop")
    print(f"   forgather control save-stop {job_id}    # Save and stop")
    print()
    print("🏃 Starting training... (use control commands to interact)")
    print("="*60)
    
    try:
        # Start training
        result = trainer.train()
        
        print()
        print("✅ Training completed successfully!")
        print(f"   Final step: {result.global_step}")
        print(f"   Final loss: {result.training_loss:.4f}")
        
    except KeyboardInterrupt:
        print()
        print("⏹️  Training interrupted by user")
        
    except Exception as e:
        print()
        print(f"❌ Training failed with error: {e}")
        raise
    
    finally:
        print()
        print("🧹 Cleaning up...")
        # Trainer cleanup happens automatically
        print("✅ Demo completed!")


if __name__ == "__main__":
    main()