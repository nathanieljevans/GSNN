"""
Test script to demonstrate the new normalization methods with small batch sizes.
This script compares different normalization approaches when using very small batches.
"""

import torch
import numpy as np
from gsnn.models.GroupLayerNorm import GroupLayerNorm
from gsnn.models.GroupBatchNorm import GroupBatchNorm
from gsnn.models.GroupRMSNorm import GroupRMSNorm
from gsnn.models.GroupEMANorm import GroupEMANorm


def test_small_batch_normalization():
    """Compare different normalization methods with small batch sizes."""
    
    # Setup
    torch.manual_seed(42)
    batch_sizes = [1, 2, 4, 8]
    num_channels = 12
    channel_groups = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]  # 4 groups with 3 channels each
    
    # Initialize all normalization methods
    norms = {
        'layer': GroupLayerNorm(channel_groups),
        'batch': torch.nn.BatchNorm1d(num_channels),
        'groupbatch': GroupBatchNorm(channel_groups, affine=True),
        'rms': GroupRMSNorm(channel_groups, affine=True),
        'ema': GroupEMANorm(channel_groups, affine=True)
    }
    
    print("Testing normalization methods with small batch sizes...")
    print("=" * 60)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 30)
        
        # Create random input
        x = torch.randn(batch_size, num_channels)
        
        # Test each normalization method
        for name, norm in norms.items():
            try:
                # For EMA norm, we need to "warm up" the running statistics
                if name == 'ema':
                    norm.train()
                    # Warm up with a few batches
                    for _ in range(10):
                        _ = norm(torch.randn(max(batch_size, 8), num_channels))
                
                norm.eval()  # Set to eval mode
                with torch.no_grad():
                    output = norm(x)
                    
                # Check for NaN or extreme values
                has_nan = torch.isnan(output).any()
                has_inf = torch.isinf(output).any()
                std_value = output.std().item()
                mean_value = output.mean().item()
                
                status = "✓ OK"
                if has_nan:
                    status = "✗ NaN detected"
                elif has_inf:
                    status = "✗ Inf detected"
                elif std_value > 10 or abs(mean_value) > 10:
                    status = "⚠ Large values"
                
                print(f"  {name:12s}: {status:15s} (mean: {mean_value:6.3f}, std: {std_value:6.3f})")
                
            except Exception as e:
                print(f"  {name:12s}: ✗ ERROR - {str(e)[:40]}...")


def test_stability_over_time():
    """Test how normalization methods behave over multiple training steps with small batches."""
    
    print("\n" + "=" * 60)
    print("Testing stability over training with batch size 2...")
    print("=" * 60)
    
    torch.manual_seed(42)
    batch_size = 2
    num_channels = 8
    channel_groups = [0, 0, 1, 1, 2, 2, 3, 3]  # 4 groups with 2 channels each
    num_steps = 100
    
    # Initialize normalization methods
    norms = {
        'rms': GroupRMSNorm(channel_groups, affine=True),
        'ema': GroupEMANorm(channel_groups, affine=True),
        'layer': GroupLayerNorm(channel_groups, affine=True)
    }
    
    for name, norm in norms.items():
        norm.train()
        outputs = []
        
        print(f"\nTesting {name} normalization:")
        for step in range(num_steps):
            x = torch.randn(batch_size, num_channels)
            output = norm(x)
            outputs.append(output.mean().item())
            
            # Print progress every 20 steps
            if (step + 1) % 20 == 0:
                recent_mean = np.mean(outputs[-20:])
                recent_std = np.std(outputs[-20:])
                print(f"  Step {step+1:3d}: mean = {recent_mean:6.3f}, std = {recent_std:6.3f}")
        
        # Final statistics
        final_mean = np.mean(outputs[-20:])
        final_std = np.std(outputs[-20:])
        print(f"  Final 20 steps: mean = {final_mean:6.3f}, std = {final_std:6.3f}")


if __name__ == "__main__":
    test_small_batch_normalization()
    test_stability_over_time()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("- RMSNorm: Simple, stable, no mean centering")
    print("- EMANorm: Uses running stats, very stable for small batches")
    print("- GroupLayerNorm: Already good for small batches (normalizes within features)")
    print("- For small batches, prefer: rms > ema > layer > groupbatch > batch")
    print("=" * 60) 