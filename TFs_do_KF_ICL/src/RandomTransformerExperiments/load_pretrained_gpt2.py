"""
Load pretrained GPT2 model from checkpoint and verify it loads successfully.
"""
import sys
import os
# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from core import Config
from models import GPT2

def load_pretrained_gpt2():
    """Load pretrained GPT2 model from checkpoint"""
    
    # Get configuration
    config = Config()
    
    # Override model_type to ensure GPT2 architecture is used
    config.override("model_type", "GPT2")
    
    # Path to the pretrained checkpoint
    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        "Set-up Data",
        "step%3D99000.ckpt"
    )
    
    print("\n" + "="*60)
    print("LOADING PRETRAINED GPT2 MODEL")
    print("="*60 + "\n")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint not found at: {checkpoint_path}")
        return None
    
    print(f"✓ Checkpoint found at: {checkpoint_path}")
    print(f"  File size: {os.path.getsize(checkpoint_path) / (1024**2):.2f} MB\n")
    
    # First create the model with correct architecture
    print("Creating model...")
    try:
        model = GPT2(
            n_dims_in=config.n_dims_in,
            n_positions=config.n_positions,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_dims_out=config.n_dims_out
        )
        print("✓ Model created successfully!\n")
    except Exception as e:
        print(f"❌ ERROR creating model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Load the checkpoint weights
    print("Loading checkpoint weights...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract the state dict from the checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load the state dict into the model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"✓ Checkpoint weights loaded successfully!")
        if missing_keys:
            print(f"  Note: {len(missing_keys)} keys in model not found in checkpoint (newly initialized)")
        if unexpected_keys:
            print(f"  Note: {len(unexpected_keys)} keys in checkpoint not in model (skipped)\n")
    except Exception as e:
        print(f"❌ ERROR loading checkpoint weights: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Verify model is loaded correctly
    print("-" * 60)
    print("MODEL INFORMATION:")
    print("-" * 60)
    
    # Print model architecture details
    print(f"Model name: {getattr(model, 'name', 'GPT2')}")
    print(f"Input dimension: {model.n_dims_in}")
    print(f"Output dimension: {model.n_dims_out}")
    print(f"Context length: {model.n_positions}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Number of layers: {config.n_layer}")
    print(f"Number of heads: {config.n_head}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Test forward pass
    print("\n" + "-" * 60)
    print("TESTING FORWARD PASS:")
    print("-" * 60)
    
    # Create dummy input
    batch_size = 2
    seq_len = config.n_positions
    dummy_input = {
        "current": torch.randn(batch_size, seq_len, config.n_dims_in),
        "target": torch.randn(batch_size, seq_len, config.n_dims_out)
    }
    
    print(f"Input shape: {dummy_input['current'].shape}")
    print(f"Target shape: {dummy_input['target'].shape}")
    
    # Set model to eval mode
    model.eval()
    
    # Forward pass
    try:
        with torch.no_grad():
            output_dict = model(dummy_input)
        
        print(f"\n✓ Forward pass successful!")
        print(f"  Loss: {output_dict['optimized_loss'].item():.4f}")
        print(f"  Loss keys: {list(output_dict.keys())}")
    except Exception as e:
        print(f"❌ ERROR during forward pass: {e}")
        return None
    
    print("\n" + "="*60)
    print("MODEL LOADED AND VERIFIED SUCCESSFULLY! ✅")
    print("="*60 + "\n")
    
    return model

if __name__ == "__main__":
    model = load_pretrained_gpt2()
    
    if model is not None:
        print("You can now use the model for inference or further training!")
        print("\nExample usage:")
        print("  model.eval()  # Set to evaluation mode")
        print("  with torch.no_grad():")
        print("      predictions = model.predict_step(input_dict)")

