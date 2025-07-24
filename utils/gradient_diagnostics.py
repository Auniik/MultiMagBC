import torch
import torch.nn as nn
from collections import defaultdict


def diagnose_gradients(model, loss, verbose=True):
    """Diagnose gradient issues by analyzing gradients per layer"""
    
    # Compute gradients
    loss.backward(retain_graph=True)
    
    gradient_stats = {}
    problematic_layers = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            
            # Calculate statistics
            grad_norm = grad.norm().item()
            grad_max = grad.abs().max().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()
            
            # Check for issues
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()
            is_large = grad_norm > 10.0
            
            gradient_stats[name] = {
                'norm': grad_norm,
                'max': grad_max,
                'mean': grad_mean,
                'std': grad_std,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'is_large': is_large
            }
            
            # Flag problematic layers
            if has_nan or has_inf or is_large:
                problematic_layers.append((name, grad_norm))
    
    if verbose and problematic_layers:
        print("🚨 Problematic layers (large gradients):")
        for name, norm in sorted(problematic_layers, key=lambda x: x[1], reverse=True):
            print(f"  {name}: {norm:.4f}")
    
    return gradient_stats, problematic_layers


def analyze_activations(model, sample_input, verbose=True):
    """Analyze activations to find saturation issues"""
    
    activation_stats = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                output_flat = output.view(-1)
                activation_stats[name] = {
                    'mean': output_flat.mean().item(),
                    'std': output_flat.std().item(),
                    'min': output_flat.min().item(),
                    'max': output_flat.max().item(),
                    'has_nan': torch.isnan(output_flat).any().item(),
                    'has_inf': torch.isinf(output_flat).any().item(),
                    'near_zero': (output_flat.abs() < 1e-6).float().mean().item(),
                    'saturated': ((output_flat.abs() > 10).float().mean().item() if output_flat.abs().max() > 0 else 0)
                }
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(**sample_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    if verbose:
        print("🔍 Activation Analysis:")
        for name, stats in activation_stats.items():
            if stats['has_nan'] or stats['has_inf'] or stats['saturated'] > 0.1:
                print(f"  ⚠️ {name}: NaN={stats['has_nan']}, Inf={stats['has_inf']}, "
                      f"Saturated={stats['saturated']:.2%}, Range=[{stats['min']:.3f}, {stats['max']:.3f}]")
    
    return activation_stats


def check_parameter_magnitudes(model, verbose=True):
    """Check if any parameters have extreme values"""
    
    param_stats = {}
    large_params = []
    
    for name, param in model.named_parameters():
        param_norm = param.norm().item()
        param_max = param.abs().max().item()
        
        param_stats[name] = {
            'norm': param_norm,
            'max': param_max,
            'mean': param.mean().item(),
            'std': param.std().item()
        }
        
        if param_norm > 100 or param_max > 50:
            large_params.append((name, param_norm, param_max))
    
    if verbose and large_params:
        print("📊 Large parameter values:")
        for name, norm, max_val in large_params:
            print(f"  {name}: norm={norm:.2f}, max={max_val:.2f}")
    
    return param_stats, large_params


def gradient_flow_analysis(model, loss, layer_types=None):
    """Analyze gradient flow through different layer types"""
    
    if layer_types is None:
        layer_types = [nn.Linear, nn.Conv2d, nn.MultiheadAttention, nn.LayerNorm]
    
    loss.backward(retain_graph=True)
    
    flow_stats = defaultdict(list)
    
    for name, module in model.named_modules():
        if any(isinstance(module, lt) for lt in layer_types):
            for param_name, param in module.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    layer_type = type(module).__name__
                    flow_stats[layer_type].append(grad_norm)
    
    # Calculate statistics per layer type
    summary = {}
    for layer_type, norms in flow_stats.items():
        if norms:
            summary[layer_type] = {
                'count': len(norms),
                'mean': sum(norms) / len(norms),
                'max': max(norms),
                'min': min(norms)
            }
    
    return summary


def comprehensive_gradient_diagnosis(model, loss, sample_input, verbose=True):
    """Run comprehensive gradient diagnostics"""
    
    print("🔬 Running comprehensive gradient diagnosis...")
    print("=" * 50)
    
    # 1. Gradient analysis
    grad_stats, problem_layers = diagnose_gradients(model, loss, verbose)
    
    print("\n" + "=" * 50)
    
    # 2. Activation analysis  
    act_stats = analyze_activations(model, sample_input, verbose)
    
    print("\n" + "=" * 50)
    
    # 3. Parameter magnitude check
    param_stats, large_params = check_parameter_magnitudes(model, verbose)
    
    print("\n" + "=" * 50)
    
    # 4. Gradient flow analysis
    flow_stats = gradient_flow_analysis(model, loss)
    if verbose:
        print("🌊 Gradient flow by layer type:")
        for layer_type, stats in flow_stats.items():
            print(f"  {layer_type}: mean={stats['mean']:.4f}, max={stats['max']:.4f}")
    
    return {
        'gradients': grad_stats,
        'activations': act_stats,
        'parameters': param_stats,
        'flow': flow_stats,
        'problem_layers': problem_layers
    }