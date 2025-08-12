import torch
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

class GradDiagnostics:
    """
    Gradient Diagnostics for Vanishing Gradient Analysis in PyTorch Models.

    Tracks and summarizes gradient vanishing effects over training steps.
    Useful for diagnosing vanishing gradient problems in deep or recurrent networks.

    Usage:
        diag = GradDiagnostics(window_size=100, verbose=True)
        # During training loop, after loss.backward():
        diag.update(model, loss, step)
        # To get the latest summary:
        summary = diag.get_summary()
        # To reset history:
        diag.reset()
    """
    def __init__(self, window_size: int = 100, verbose: bool = True):
        """
        Args:
            window_size (int): Number of recent steps to keep in history.
            verbose (bool): If True, print warnings for high vanishing fraction.
        """
        self.window_size = window_size
        self.verbose = verbose
        self.history = []  # List of per-step summaries
        self.step = 0

    def analyze(self, model: torch.nn.Module, threshold: float = 1e-6) -> Dict[str, Any]:
        """
        Analyze and summarize gradient vanishing effects in the model.

        For each parameter tensor in the model, computes the mean and maximum absolute gradient
        after a backward pass, and the fraction of elements with very small gradients (|grad| < threshold).
        If a large fraction of parameters have near-zero gradients, it may indicate vanishing gradients.

        Args:
            model (torch.nn.Module): Model to analyze. Should have gradients computed (after backward).
            threshold (float): The absolute gradient value below which a gradient is considered 'vanished'.

        Returns:
            dict: A dictionary containing per-parameter statistics and an overall summary, including:
                - 'per_layer': {layer_name: {'mean_abs_grad', 'max_abs_grad', 'vanishing_frac'}}
                - 'overall': {'avg_vanishing_frac', 'layers_with_high_vanishing', 'threshold'}
        """
        grad_stats = {}
        vanishing_fracs = []
        layers_with_high_vanishing = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().abs().cpu().view(-1)
                mean_abs_grad = grad.mean().item()
                max_abs_grad = grad.max().item()
                vanishing_frac = (grad < threshold).float().mean().item()
                grad_stats[name] = {
                    'mean_abs_grad': mean_abs_grad,
                    'max_abs_grad': max_abs_grad,
                    'vanishing_frac': vanishing_frac
                }
                vanishing_fracs.append(vanishing_frac)
                if vanishing_frac > 0.99:
                    layers_with_high_vanishing.append(name)
            else:
                grad_stats[name] = {
                    'mean_abs_grad': None,
                    'max_abs_grad': None,
                    'vanishing_frac': None
                }
        avg_vanishing_frac = float(np.mean(vanishing_fracs)) if vanishing_fracs else None
        summary = {
            'per_layer': grad_stats,
            'overall': {
                'avg_vanishing_frac': avg_vanishing_frac,
                'layers_with_high_vanishing': layers_with_high_vanishing,
                'threshold': threshold
            }
        }
        if self.verbose and layers_with_high_vanishing:
            print(f"[GradDiagnostics] Warning: Layers with >99% vanishing gradients: {layers_with_high_vanishing}")
        return summary

    def update(self, model: torch.nn.Module, loss: float, step: Optional[int] = None, threshold: float = 1e-6):
        """
        Update diagnostics with the current model gradients and loss.
        Stores the latest summary in the history (rolling window), along with step and loss.

        Args:
            model (torch.nn.Module): Model to analyze. Should have gradients computed (after backward).
            loss (float): Current loss value.
            step (int, optional): Step counter. If None, increments internal step.
            threshold (float): Threshold for vanishing gradient detection.
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1
        summary = self.analyze(model, threshold=threshold)
        record = {
            'step': self.step,
            'loss': loss,
            'grad_summary': summary
        }
        self.history.append(record)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get the most recent gradient vanishing summary, including step and loss.

        Returns:
            dict: The latest summary dictionary, or None if no history.
        """
        if self.history:
            return self.history[-1]
        else:
            return None

    def reset(self):
        """
        Reset the diagnostic history and step counter.
        """
        self.history.clear()
        self.step = 0

    def plot_diagnostics(self, step: int = None):
        """
        Plot the vanishing gradient fraction by layer number for different parameter types.
        X-axis: layer number (extracted from parameter names like 'ResBlocks.0.lin_in.weight')
        Y-axis: vanishing gradient fraction (|grad| < threshold) for that step.
        Different parameter types (lin_in, lin_out, norm, etc.) are shown as different colored lines.

        Args:
            step (int, optional): Index of the step in history to plot. If None, uses the most recent step.
        """
        if not self.history:
            print("No diagnostic history to plot.")
            return
        
        # Determine which record to plot
        if step is None:
            record = self.history[-1]
            step_label = record['step']
        else:
            if step < 0 or step >= len(self.history):
                print(f"Step {step} out of range, using most recent step.")
                record = self.history[-1]
                step_label = record['step']
            else:
                record = self.history[step]
                step_label = record['step']
        
        grad_summary = record['grad_summary']
        
        # Parse parameter names to extract layer numbers and types
        import re
        layer_data = {}  # {param_type: {layer_num: vanishing_frac}}
        
        for param_name, stats in grad_summary['per_layer'].items():
            vanishing_frac = stats['vanishing_frac']
            if vanishing_frac is None:
                continue
                
            # Extract layer number (e.g., "ResBlocks.0.lin_in.weight" -> layer 0)
            layer_match = re.search(r'ResBlocks\.(\d+)', param_name)
            if not layer_match:
                continue
            layer_num = int(layer_match.group(1))
            
            # Extract parameter type (lin_in, lin_out, norm, etc.)
            if 'lin_in' in param_name:
                param_type = 'lin_in'
            elif 'lin_out' in param_name:
                param_type = 'lin_out'
            elif 'norm' in param_name:
                param_type = 'norm'
            elif 'nonlin' in param_name:
                param_type = 'nonlin'
            elif 'residual_weight' in param_name:
                param_type = 'residual_weight'
            else:
                param_type = 'other'
            
            if param_type not in layer_data:
                layer_data[param_type] = {}
            layer_data[param_type][layer_num] = vanishing_frac
        
        # Plot
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(layer_data)))
        
        for i, (param_type, layer_fracs) in enumerate(layer_data.items()):
            layers = sorted(layer_fracs.keys())
            fracs = [layer_fracs[layer] for layer in layers]
            plt.plot(layers, fracs, marker='o', label=param_type, color=colors[i])
        
        plt.xlabel('Layer')
        plt.ylabel('Vanishing gradient fraction (|grad| < threshold)')
        plt.title(f'Vanishing Gradient Fraction by Layer (Step {step_label})')
        plt.legend(title='Parameter Type', loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show() 

    def plot_gradient_magnitude_by_layer(self, step: int = None):
        """
        Plot the mean absolute gradient magnitude by layer number for different parameter types.
        X-axis: layer number, Y-axis: mean absolute gradient magnitude.
        Different parameter types shown as different colored lines.

        Args:
            step (int, optional): Index of the step in history to plot. If None, uses the most recent step.
        """
        if not self.history:
            print("No diagnostic history to plot.")
            return
        
        # Determine which record to plot
        if step is None:
            record = self.history[-1]
            step_label = record['step']
        else:
            if step < 0 or step >= len(self.history):
                print(f"Step {step} out of range, using most recent step.")
                record = self.history[-1]
                step_label = record['step']
            else:
                record = self.history[step]
                step_label = record['step']
        
        grad_summary = record['grad_summary']
        
        # Parse parameter names to extract layer numbers and types
        import re
        layer_data = {}  # {param_type: {layer_num: mean_abs_grad}}
        
        for param_name, stats in grad_summary['per_layer'].items():
            mean_abs_grad = stats['mean_abs_grad']
            if mean_abs_grad is None:
                continue
                
            # Extract layer number
            layer_match = re.search(r'ResBlocks\.(\d+)', param_name)
            if not layer_match:
                continue
            layer_num = int(layer_match.group(1))
            
            # Extract parameter type
            if 'lin_in' in param_name:
                param_type = 'lin_in'
            elif 'lin_out' in param_name:
                param_type = 'lin_out'
            elif 'norm' in param_name:
                param_type = 'norm'
            elif 'nonlin' in param_name:
                param_type = 'nonlin'
            elif 'residual_weight' in param_name:
                param_type = 'residual_weight'
            else:
                param_type = 'other'
            
            if param_type not in layer_data:
                layer_data[param_type] = {}
            layer_data[param_type][layer_num] = mean_abs_grad
        
        # Plot
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(layer_data)))
        
        for i, (param_type, layer_grads) in enumerate(layer_data.items()):
            layers = sorted(layer_grads.keys())
            grads = [layer_grads[layer] for layer in layers]
            plt.semilogy(layers, grads, marker='o', label=param_type, color=colors[i])
        
        plt.xlabel('Layer')
        plt.ylabel('Mean Absolute Gradient Magnitude (log scale)')
        plt.title(f'Gradient Magnitude by Layer (Step {step_label})')
        plt.legend(title='Parameter Type', loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_vanishing_over_time(self, layers_to_show: list = None):
        """
        Plot vanishing gradient fraction over time for specific layers.
        X-axis: training step, Y-axis: vanishing gradient fraction.
        Different layers shown as different colored lines.

        Args:
            layers_to_show (list, optional): List of layer numbers to show. If None, shows first 5 layers.
        """
        if not self.history:
            print("No diagnostic history to plot.")
            return
        
        if layers_to_show is None:
            layers_to_show = [0, 1, 2, 3, 4]  # Default to first 5 layers
        
        # Collect data over time
        import re
        steps = []
        layer_data = {layer: [] for layer in layers_to_show}
        
        for record in self.history:
            steps.append(record['step'])
            grad_summary = record['grad_summary']
            
            # Initialize layer values for this step
            layer_values = {layer: None for layer in layers_to_show}
            
            for param_name, stats in grad_summary['per_layer'].items():
                vanishing_frac = stats['vanishing_frac']
                if vanishing_frac is None:
                    continue
                
                # Extract layer number
                layer_match = re.search(r'ResBlocks\.(\d+)', param_name)
                if not layer_match:
                    continue
                layer_num = int(layer_match.group(1))
                
                if layer_num in layers_to_show and 'lin_in' in param_name:  # Use lin_in as representative
                    layer_values[layer_num] = vanishing_frac
            
            # Append values for each layer
            for layer in layers_to_show:
                layer_data[layer].append(layer_values[layer])
        
        # Plot
        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers_to_show)))
        
        for i, layer in enumerate(layers_to_show):
            valid_indices = [j for j, val in enumerate(layer_data[layer]) if val is not None]
            valid_steps = [steps[j] for j in valid_indices]
            valid_values = [layer_data[layer][j] for j in valid_indices]
            
            if valid_values:
                plt.plot(valid_steps, valid_values, marker='o', label=f'Layer {layer}', 
                        color=colors[i], alpha=0.8)
        
        plt.xlabel('Training Step')
        plt.ylabel('Vanishing Gradient Fraction')
        plt.title('Vanishing Gradients Over Time by Layer')
        plt.legend(title='Layer', loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_gradient_ratio_heatmap(self, steps_to_show: int = 10):
        """
        Plot a heatmap showing vanishing gradient fraction across layers and recent training steps.
        X-axis: training step, Y-axis: layer number, Color: vanishing fraction.

        Args:
            steps_to_show (int): Number of recent steps to include in heatmap.
        """
        if not self.history:
            print("No diagnostic history to plot.")
            return
        
        # Get recent steps
        recent_history = self.history[-steps_to_show:]
        if len(recent_history) < 2:
            print("Not enough history for heatmap.")
            return
        
        # Collect layer numbers and steps
        import re
        all_layers = set()
        steps = []
        
        for record in recent_history:
            steps.append(record['step'])
            grad_summary = record['grad_summary']
            
            for param_name in grad_summary['per_layer'].keys():
                layer_match = re.search(r'ResBlocks\.(\d+)', param_name)
                if layer_match and 'lin_in' in param_name:  # Use lin_in as representative
                    all_layers.add(int(layer_match.group(1)))
        
        layers = sorted(all_layers)
        
        # Build heatmap data
        heatmap_data = np.full((len(layers), len(steps)), np.nan)
        
        for j, record in enumerate(recent_history):
            grad_summary = record['grad_summary']
            
            for param_name, stats in grad_summary['per_layer'].items():
                vanishing_frac = stats['vanishing_frac']
                if vanishing_frac is None:
                    continue
                
                layer_match = re.search(r'ResBlocks\.(\d+)', param_name)
                if layer_match and 'lin_in' in param_name:
                    layer_num = int(layer_match.group(1))
                    if layer_num in layers:
                        layer_idx = layers.index(layer_num)
                        heatmap_data[layer_idx, j] = vanishing_frac
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        im = plt.imshow(heatmap_data, cmap='Reds', aspect='auto', interpolation='nearest')
        
        plt.xlabel('Training Step')
        plt.ylabel('Layer Number')
        plt.title('Vanishing Gradient Fraction Heatmap')
        
        # Set ticks
        plt.xticks(range(len(steps)), steps, rotation=45)
        plt.yticks(range(len(layers)), layers)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Vanishing Gradient Fraction')
        
        plt.tight_layout()
        plt.show()

    def plot_summary_statistics(self, step: int = None):
        """
        Plot summary statistics: mean gradient magnitude, max gradient magnitude, and vanishing fraction
        across all parameters for a given step.

        Args:
            step (int, optional): Index of the step in history to plot. If None, uses the most recent step.
        """
        if not self.history:
            print("No diagnostic history to plot.")
            return
        
        # Determine which record to plot
        if step is None:
            record = self.history[-1]
            step_label = record['step']
        else:
            if step < 0 or step >= len(self.history):
                print(f"Step {step} out of range, using most recent step.")
                record = self.history[-1]
                step_label = record['step']
            else:
                record = self.history[step]
                step_label = record['step']
        
        grad_summary = record['grad_summary']
        
        # Collect statistics by parameter type
        param_stats = {}
        
        for param_name, stats in grad_summary['per_layer'].items():
            # Extract parameter type
            if 'lin_in' in param_name:
                param_type = 'lin_in'
            elif 'lin_out' in param_name:
                param_type = 'lin_out'
            elif 'norm' in param_name:
                param_type = 'norm'
            elif 'residual_weight' in param_name:
                param_type = 'residual_weight'
            else:
                param_type = 'other'
            
            if param_type not in param_stats:
                param_stats[param_type] = {'mean_grads': [], 'max_grads': [], 'vanishing_fracs': []}
            
            if stats['mean_abs_grad'] is not None:
                param_stats[param_type]['mean_grads'].append(stats['mean_abs_grad'])
                param_stats[param_type]['max_grads'].append(stats['max_abs_grad'])
                param_stats[param_type]['vanishing_fracs'].append(stats['vanishing_frac'])
        
        # Create subplot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        param_types = list(param_stats.keys())
        x = np.arange(len(param_types))
        
        # Mean gradient magnitude
        mean_vals = [np.mean(param_stats[pt]['mean_grads']) if param_stats[pt]['mean_grads'] else 0 
                     for pt in param_types]
        axes[0].bar(x, mean_vals)
        axes[0].set_xlabel('Parameter Type')
        axes[0].set_ylabel('Mean Abs Gradient')
        axes[0].set_title('Mean Gradient Magnitude')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(param_types, rotation=45)
        axes[0].set_yscale('log')
        
        # Max gradient magnitude
        max_vals = [np.mean(param_stats[pt]['max_grads']) if param_stats[pt]['max_grads'] else 0 
                    for pt in param_types]
        axes[1].bar(x, max_vals)
        axes[1].set_xlabel('Parameter Type')
        axes[1].set_ylabel('Mean Max Gradient')
        axes[1].set_title('Max Gradient Magnitude')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(param_types, rotation=45)
        axes[1].set_yscale('log')
        
        # Vanishing fraction
        vanishing_vals = [np.mean(param_stats[pt]['vanishing_fracs']) if param_stats[pt]['vanishing_fracs'] else 0 
                          for pt in param_types]
        axes[2].bar(x, vanishing_vals)
        axes[2].set_xlabel('Parameter Type')
        axes[2].set_ylabel('Vanishing Fraction')
        axes[2].set_title('Vanishing Gradient Fraction')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(param_types, rotation=45)
        
        plt.suptitle(f'Gradient Summary Statistics (Step {step_label})')
        plt.tight_layout()
        plt.show() 