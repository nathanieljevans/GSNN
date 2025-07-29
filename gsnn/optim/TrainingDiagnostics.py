import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
import warnings
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import copy


class TrainingDiagnostics:
    """
    Comprehensive training diagnostics for monitoring model optimization.
    
    Tracks gradient flow, activation patterns, weight spectral properties,
    and basic curvature metrics to help diagnose training issues and guide
    optimization decisions.
    
    Usage:
        diagnostics = TrainingDiagnostics(model, track_every=10)
        
        # During training loop:
        loss.backward()
        diagnostics.update(model, loss.item(), batch_idx)
        optimizer.step()
        
        # Generate reports:
        summary = diagnostics.get_summary()
        diagnostics.plot_diagnostics()
    """
    
    def __init__(self, model: nn.Module, track_every: int = 10, 
                 window_size: int = 100, track_activations: bool = True,
                 track_weights: bool = True, track_gradients: bool = True,
                 track_curvature: bool = False, verbose: bool = True):
        """
        Initialize training diagnostics tracker.
        
        Args:
            model: PyTorch model to monitor
            track_every: Update diagnostics every N steps
            window_size: Size of rolling window for statistics
            track_activations: Whether to track activation statistics
            track_weights: Whether to track weight spectral properties
            track_gradients: Whether to track gradient flow metrics
            track_curvature: Whether to track curvature (expensive!)
            verbose: Whether to print diagnostic warnings
        """
        self.model = model
        self.track_every = track_every
        self.window_size = window_size
        self.track_activations = track_activations
        self.track_weights = track_weights  
        self.track_gradients = track_gradients
        self.track_curvature = track_curvature
        self.verbose = verbose
        
        # Storage for metrics
        self.step = 0
        self.losses = deque(maxlen=window_size)
        
        # Gradient flow diagnostics
        self.grad_norms = defaultdict(list)
        self.weight_norms = defaultdict(list)
        self.grad_weight_ratios = defaultdict(list)
        self.grad_noise_scale = []
        self.grad_cosine_sim = []
        
        # Activation diagnostics  
        self.dead_neuron_rates = defaultdict(list)
        self.activation_saturations = defaultdict(list)
        self.feature_variance = defaultdict(list)
        
        # Weight spectral diagnostics
        self.weight_singular_values = defaultdict(list)
        self.condition_numbers = defaultdict(list)
        
        # Curvature diagnostics
        self.hessian_traces = []
        self.sharpness_estimates = []
        
        # Activation hooks
        self.activation_hooks = []
        self.activations = {}
        
        if self.track_activations:
            self._register_activation_hooks()
            
        # Store previous gradients for noise computation
        self.prev_gradients = None
        
    def _register_activation_hooks(self):
        """Register forward hooks to capture activations."""
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach()
                elif isinstance(output, (list, tuple)) and len(output) > 0:
                    self.activations[name] = output[0].detach()
            return hook
            
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                handle = module.register_forward_hook(make_hook(name))
                self.activation_hooks.append(handle)
    
    def update(self, model: nn.Module, loss: float, step: Optional[int] = None):
        """
        Update diagnostics with current training state.
        
        Args:
            model: Current model state
            loss: Current loss value
            step: Optional step counter (uses internal if None)
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1
            
        self.losses.append(loss)
        
        # Only compute expensive diagnostics every N steps
        if self.step % self.track_every != 0:
            return
            
        with torch.no_grad():
            if self.track_gradients:
                self._update_gradient_diagnostics(model)
                
            if self.track_activations:
                self._update_activation_diagnostics()
                
            if self.track_weights:
                self._update_weight_diagnostics(model)
                
            if self.track_curvature:
                self._update_curvature_diagnostics(model)
    
    def _update_gradient_diagnostics(self, model: nn.Module):
        """Update gradient flow diagnostics."""
        gradients = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                weight = param.data
                
                # Gradient and weight norms
                grad_norm = torch.norm(grad).item()
                weight_norm = torch.norm(weight).item()
                
                self.grad_norms[name].append(grad_norm)
                self.weight_norms[name].append(weight_norm)
                
                # Gradient/weight ratio
                ratio = grad_norm / (weight_norm + 1e-8)
                self.grad_weight_ratios[name].append(ratio)
                
                # Collect gradients for noise analysis
                gradients.append(grad.flatten())
                
                # Check for problematic gradients
                if self.verbose:
                    if grad_norm > 10.0:
                        warnings.warn(f"Large gradient norm in {name}: {grad_norm:.4f}")
                    if ratio > 1.0:
                        warnings.warn(f"Large grad/weight ratio in {name}: {ratio:.4f}")
        
        # Gradient noise scale computation
        if len(gradients) > 0:
            grad_vec = torch.cat(gradients)
            
            if self.prev_gradients is not None:
                # Compute gradient noise as variance/mean^2
                grad_diff = grad_vec - self.prev_gradients
                noise = torch.var(grad_diff) / (torch.mean(grad_vec)**2 + 1e-8)
                self.grad_noise_scale.append(noise.item())
                
                # Gradient cosine similarity
                cos_sim = torch.cosine_similarity(grad_vec.unsqueeze(0), 
                                                self.prev_gradients.unsqueeze(0))
                self.grad_cosine_sim.append(cos_sim.item())
            
            self.prev_gradients = grad_vec.clone()
    
    def _update_activation_diagnostics(self):
        """Update activation space diagnostics."""
        for name, activation in self.activations.items():
            if activation.numel() == 0:
                continue
                
            # Dead neuron rate (for ReLU-like activations)
            dead_rate = (activation <= 0).float().mean().item()
            self.dead_neuron_rates[name].append(dead_rate)
            
            # Activation saturation (for sigmoid/tanh)
            if activation.min() >= 0 and activation.max() <= 1:  # Likely sigmoid
                saturation = ((activation < 0.01) | (activation > 0.99)).float().mean().item()
                self.activation_saturations[name].append(saturation)
            elif activation.min() >= -1 and activation.max() <= 1:  # Likely tanh
                saturation = ((activation < -0.99) | (activation > 0.99)).float().mean().item()
                self.activation_saturations[name].append(saturation)
            
            # Feature variance across batch (detect feature collapse)
            if len(activation.shape) >= 2:
                feature_var = activation.var(dim=0).mean().item()
                self.feature_variance[name].append(feature_var)
        
        # Clear activations for next iteration
        self.activations.clear()
    
    def _update_weight_diagnostics(self, model: nn.Module):
        """Update weight spectral diagnostics."""
        for name, param in model.named_parameters():
            if len(param.shape) >= 2:  # Only for weight matrices
                weight = param.data
                
                # Compute singular values
                try:
                    U, S, V = torch.svd(weight.flatten(1))
                    singular_vals = S.cpu().numpy()
                    
                    self.weight_singular_values[name].append({
                        'max': float(singular_vals.max()),
                        'min': float(singular_vals[singular_vals > 1e-8].min()) if len(singular_vals[singular_vals > 1e-8]) > 0 else 1e-8,
                        'mean': float(singular_vals.mean()),
                        'std': float(singular_vals.std())
                    })
                    
                    # Condition number
                    cond_num = float(singular_vals.max() / (singular_vals[singular_vals > 1e-8].min() + 1e-8))
                    self.condition_numbers[name].append(cond_num)
                    
                    # Check for problematic spectra
                    if self.verbose:
                        if cond_num > 1000:
                            warnings.warn(f"High condition number in {name}: {cond_num:.2f}")
                        if singular_vals.max() > 10:
                            warnings.warn(f"Large singular values in {name}: max={singular_vals.max():.2f}")
                            
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"SVD failed for {name}: {e}")
    
    def _update_curvature_diagnostics(self, model: nn.Module):
        """Update curvature diagnostics (expensive!)."""
        try:
            # Hutchinson trace estimator for Hessian trace
            params = list(model.parameters())
            z = torch.randn_like(parameters_to_vector(params))
            
            # This is a simplified version - in practice you'd need the loss function
            # For now, just store a placeholder
            self.hessian_traces.append(0.0)
            
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Curvature computation failed: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive diagnostic summary.
        
        Returns:
            Dictionary containing current diagnostic state and recommendations
        """
        summary = {
            'step': self.step,
            'current_loss': self.losses[-1] if self.losses else None,
            'gradient_health': {},
            'activation_health': {},
            'weight_health': {},
            'recommendations': []
        }
        
        # Gradient health summary
        if self.grad_weight_ratios:
            latest_ratios = {name: vals[-1] for name, vals in self.grad_weight_ratios.items() if vals}
            avg_ratio = np.mean(list(latest_ratios.values()))
            summary['gradient_health'] = {
                'avg_grad_weight_ratio': avg_ratio,
                'problematic_layers': [name for name, ratio in latest_ratios.items() if ratio > 1.0],
                'gradient_noise': self.grad_noise_scale[-1] if self.grad_noise_scale else None,
                'gradient_alignment': self.grad_cosine_sim[-1] if self.grad_cosine_sim else None
            }
            
            # Generate recommendations
            if avg_ratio > 1.0:
                summary['recommendations'].append("High gradient/weight ratios detected. Consider gradient clipping or learning rate reduction.")
            if self.grad_noise_scale and self.grad_noise_scale[-1] > 1.0:
                summary['recommendations'].append("High gradient noise. Consider increasing batch size.")
            if self.grad_cosine_sim and self.grad_cosine_sim[-1] < 0.1:
                summary['recommendations'].append("Low gradient alignment. Training may be unstable.")
        
        # Activation health summary
        if self.dead_neuron_rates:
            latest_dead_rates = {name: vals[-1] for name, vals in self.dead_neuron_rates.items() if vals}
            avg_dead_rate = np.mean(list(latest_dead_rates.values()))
            summary['activation_health'] = {
                'avg_dead_neuron_rate': avg_dead_rate,
                'layers_with_dead_neurons': [name for name, rate in latest_dead_rates.items() if rate > 0.5]
            }
            
            if avg_dead_rate > 0.5:
                summary['recommendations'].append("High dead neuron rate. Consider LeakyReLU, better initialization, or lower learning rates.")
        
        # Weight health summary
        if self.condition_numbers:
            latest_cond_nums = {name: vals[-1] for name, vals in self.condition_numbers.items() if vals}
            avg_cond_num = np.mean(list(latest_cond_nums.values()))
            summary['weight_health'] = {
                'avg_condition_number': avg_cond_num,
                'ill_conditioned_layers': [name for name, cond in latest_cond_nums.items() if cond > 1000]
            }
            
            if avg_cond_num > 1000:
                summary['recommendations'].append("High condition numbers detected. Consider spectral normalization or better initialization.")
        
        return summary
    
    def plot_diagnostics(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 12)):
        """
        Generate comprehensive diagnostic plots.
        
        Args:
            save_path: Optional path to save the plot
            figsize: Figure size tuple
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle(f'Training Diagnostics (Step {self.step})', fontsize=16)
        
        # Loss curve
        if self.losses:
            axes[0, 0].plot(list(self.losses))
            axes[0, 0].set_title('Loss Curve')
            axes[0, 0].set_xlabel('Recent Steps')
            axes[0, 0].set_ylabel('Loss')
        
        # Gradient/weight ratios
        if self.grad_weight_ratios:
            for name, ratios in self.grad_weight_ratios.items():
                axes[0, 1].plot(ratios[-min(50, len(ratios)):], label=name[:15], alpha=0.7)
            axes[0, 1].set_title('Gradient/Weight Ratios')
            axes[0, 1].set_ylabel('Ratio')
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        
        # Gradient noise scale
        if self.grad_noise_scale:
            axes[0, 2].plot(self.grad_noise_scale)
            axes[0, 2].set_title('Gradient Noise Scale')
            axes[0, 2].set_ylabel('Noise Scale')
        
        # Dead neuron rates
        if self.dead_neuron_rates:
            for name, rates in self.dead_neuron_rates.items():
                axes[1, 0].plot(rates[-min(50, len(rates)):], label=name[:15], alpha=0.7)
            axes[1, 0].set_title('Dead Neuron Rates')
            axes[1, 0].set_ylabel('Dead Rate')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Condition numbers
        if self.condition_numbers:
            for name, conds in self.condition_numbers.items():
                axes[1, 1].semilogy(conds[-min(50, len(conds)):], label=name[:15], alpha=0.7)
            axes[1, 1].set_title('Condition Numbers')
            axes[1, 1].set_ylabel('Condition Number')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Feature variance (collapse detection)
        if self.feature_variance:
            for name, variances in self.feature_variance.items():
                axes[1, 2].plot(variances[-min(50, len(variances)):], label=name[:15], alpha=0.7)
            axes[1, 2].set_title('Feature Variance')
            axes[1, 2].set_ylabel('Variance')
            axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Gradient cosine similarity
        if self.grad_cosine_sim:
            axes[2, 0].plot(self.grad_cosine_sim)
            axes[2, 0].set_title('Gradient Cosine Similarity')
            axes[2, 0].set_ylabel('Cosine Similarity')
            axes[2, 0].axhline(y=0.1, color='r', linestyle='--', alpha=0.5)
        
        # Weight spectrum overview (latest)
        if self.weight_singular_values:
            layer_names = []
            max_vals = []
            min_vals = []
            for name, sv_list in self.weight_singular_values.items():
                if sv_list:
                    layer_names.append(name[:10])
                    max_vals.append(sv_list[-1]['max'])
                    min_vals.append(sv_list[-1]['min'])
            
            if layer_names:
                x_pos = np.arange(len(layer_names))
                axes[2, 1].bar(x_pos, max_vals, alpha=0.7, label='Max SV')
                axes[2, 1].bar(x_pos, min_vals, alpha=0.7, label='Min SV')
                axes[2, 1].set_title('Weight Singular Values')
                axes[2, 1].set_xticks(x_pos)
                axes[2, 1].set_xticklabels(layer_names, rotation=45)
                axes[2, 1].legend()
        
        # Activation saturation
        if self.activation_saturations:
            for name, sats in self.activation_saturations.items():
                axes[2, 2].plot(sats[-min(50, len(sats)):], label=name[:15], alpha=0.7)
            axes[2, 2].set_title('Activation Saturation')
            axes[2, 2].set_ylabel('Saturation Rate')
            axes[2, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def get_gradient_histogram_data(self) -> Dict[str, np.ndarray]:
        """
        Get gradient histograms for current step.
        
        Returns:
            Dictionary mapping layer names to gradient histograms
        """
        histograms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_flat = param.grad.data.flatten().cpu().numpy()
                histograms[name] = grad_flat
        return histograms
    
    def reset(self):
        """Reset all diagnostic tracking."""
        self.step = 0
        self.losses.clear()
        
        for attr_name in ['grad_norms', 'weight_norms', 'grad_weight_ratios',
                         'dead_neuron_rates', 'activation_saturations', 'feature_variance',
                         'weight_singular_values', 'condition_numbers']:
            getattr(self, attr_name).clear()
        
        for attr_name in ['grad_noise_scale', 'grad_cosine_sim', 'hessian_traces', 'sharpness_estimates']:
            getattr(self, attr_name).clear()
    
    def __del__(self):
        """Cleanup activation hooks."""
        for handle in self.activation_hooks:
            handle.remove() 