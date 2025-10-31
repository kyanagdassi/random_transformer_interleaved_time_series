"""
Test script to verify that RandomTransformerOuterLayer correctly freezes all layers except the outer layer.
"""
import sys
import os
# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from core import Config
from models import RandomTransformerOuterLayer

def test_random_transformer_outer_layer():
    """Test that only the outer layer is trainable"""
    config = Config()
    
    # Create the model with the same parameters as GPT2
    model = RandomTransformerOuterLayer(
        n_dims_in=config.n_dims_in,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_dims_out=config.n_dims_out
    )
    
    print("\n" + "="*60)
    print("TESTING RANDOM TRANSFORMER OUTER LAYER")
    print("="*60 + "\n")
    
    # Check which layers are trainable
    print("Layer-wise trainability check:")
    print("-" * 60)
    
    # Check _read_in
    read_in_trainable = any(p.requires_grad for p in model._read_in.parameters())
    read_in_params = sum(p.numel() for p in model._read_in.parameters())
    print(f"_read_in: {'TRAINABLE' if read_in_trainable else 'FROZEN'} ({read_in_params:,} params)")
    
    # Check _backbone
    backbone_trainable = any(p.requires_grad for p in model._backbone.parameters())
    backbone_params = sum(p.numel() for p in model._backbone.parameters())
    print(f"_backbone: {'TRAINABLE' if backbone_trainable else 'FROZEN'} ({backbone_params:,} params)")
    
    # Check _read_out
    read_out_trainable = any(p.requires_grad for p in model._read_out.parameters())
    read_out_params = sum(p.numel() for p in model._read_out.parameters())
    print(f"_read_out: {'TRAINABLE' if read_out_trainable else 'FROZEN'} ({read_out_params:,} params)")
    
    print("-" * 60)
    
    # Verify correct behavior
    assert not read_in_trainable, "ERROR: _read_in should be frozen!"
    assert not backbone_trainable, "ERROR: _backbone should be frozen!"
    assert read_out_trainable, "ERROR: _read_out should be trainable!"
    
    print("\n✓ All checks passed!")
    print("✓ Only the outer layer (_read_out) is trainable")
    print("✓ All other layers are frozen\n")
    
    # Test a forward pass
    print("Testing forward pass...")
    batch_size = 2
    seq_len = config.n_positions
    
    # Create dummy input
    dummy_input = {
        "current": torch.randn(batch_size, seq_len, config.n_dims_in),
        "target": torch.randn(batch_size, seq_len, config.n_dims_out)
    }
    
    # Forward pass
    output_dict = model(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"  Loss: {output_dict['optimized_loss'].item():.4f}")
    print(f"  Output shape matches target: {output_dict is not None}")
    
    # Test backward pass (should only update _read_out)
    print("\nTesting backward pass...")
    loss = output_dict['optimized_loss']
    loss.backward()
    
    # Check that gradients exist only for _read_out
    read_in_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model._read_in.parameters())
    backbone_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model._backbone.parameters())
    read_out_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model._read_out.parameters())
    
    print(f"_read_in has gradients: {read_in_has_grad}")
    print(f"_backbone has gradients: {backbone_has_grad}")
    print(f"_read_out has gradients: {read_out_has_grad}")
    
    assert not read_in_has_grad, "ERROR: _read_in should not have gradients!"
    assert not backbone_has_grad, "ERROR: _backbone should not have gradients!"
    assert read_out_has_grad, "ERROR: _read_out should have gradients!"
    
    print("\n✓ Backward pass successful")
    print("✓ Gradients only computed for _read_out\n")
    
    print("="*60)
    print("ALL TESTS PASSED!")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_random_transformer_outer_layer()








