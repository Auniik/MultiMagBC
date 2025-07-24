import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = {}

    def get_cam(self, images_dict, target_class=None, use_tumor_head=False):
        self.model.eval()
        self.activations.clear()

        # Register hooks for scale3
        handles = []
        for mag in ['40', '100', '200', '400']:
            layer = getattr(self.model.extractors[f'extractor_{mag}x'], 'conv_head')

            def forward_hook(module, input, output, name=mag):
                self.activations[name] = output
            handles.append(layer.register_forward_hook(forward_hook))

        # Enable gradients
        images_dict_grad = {k: v.clone().detach().requires_grad_(True).to(v.device) for k, v in images_dict.items()}

        # Forward
        outputs = self.model(images_dict_grad)
        if isinstance(outputs, tuple):
            logits = outputs[1] if use_tumor_head else outputs[0]
        else:
            logits = outputs

        # Ensure logits is 2D for binary classification [batch_size, 2] or [batch_size, 1]
        if len(logits.shape) == 1:
            # Single output - reshape to [batch_size, 1]
            logits = logits.unsqueeze(-1)
            # For binary classification, we need to create [batch_size, 2]
            logits = torch.cat([1 - torch.sigmoid(logits), torch.sigmoid(logits)], dim=1)

        if target_class is None:
            target_class = logits.argmax(dim=1)

        self.model.zero_grad()
        # Handle target_class properly
        if isinstance(target_class, torch.Tensor):
            target_class = target_class.item()
        
        # Ensure we have proper indexing
        class_score = logits[0, target_class] if logits.shape[0] == 1 else logits[:, target_class].sum() 

        grads = {}
        for mag in ['40', '100', '200', '400']:
            grads[mag] = torch.autograd.grad(
                class_score,
                self.activations[mag],
                retain_graph=True
            )[0]

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Generate CAMs
        cams = {}
        for mag in ['40', '100', '200', '400']:
            act = self.activations[mag]
            grad = grads[mag]
            weights = grad.mean(dim=[2, 3], keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
            cam -= cam.min()
            cam /= cam.max() + 1e-8
            cams[f'mag_{mag}'] = cam
        return cams

# === Denormalization ===
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone().detach()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

# === Visualization Function ===
def visualize_gradcam(cam_dict, images_dict, true_label=None, pred_label=None, save_path=None, show=True):
    mags = ['40', '100', '200', '400']
    fig, axes = plt.subplots(1, 8, figsize=(20, 4))

    for i, mag in enumerate(mags):
        mag_key = f'mag_{mag}'
        img = denormalize(images_dict[mag_key][0].cpu())
        cam = cam_dict[mag_key]

        # Convert to image
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)

        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap = cv2.applyColorMap((255 * cam_resized).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        # Raw image
        axes[2*i].imshow(img_np)
        axes[2*i].set_title(f'{mag}x Image')
        axes[2*i].axis('off')

        # Grad-CAM
        axes[2*i + 1].imshow(overlay[..., ::-1])  # BGR to RGB
        title = f'{mag}x CAM'
        if true_label is not None and pred_label is not None:
            correct = '[✓]' if true_label == pred_label else '[X]'
            title += f' ({correct})'
        axes[2*i + 1].set_title(title)
        axes[2*i + 1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_and_save_gradcam(model, val_loader, device, fold):
    model.eval()
    gradcam = GradCAM(model)
    for i, batch in enumerate(val_loader):
        if i >= 3: break
        sample_images = {k: v[0:1].to(device) for k, v in batch['images'].items()}
        true_label = batch['class_label'][0].item()
        with torch.no_grad():
            logits = model(sample_images)
            pred_label = logits.argmax(dim=1).item()
    
        cams = gradcam.get_cam(sample_images, target_class=pred_label)
        visualize_gradcam(
            cams,
            sample_images,
            true_label=true_label,
            pred_label=pred_label,
            save_path=f'outputs/gradcam_outputs/fold_{fold}_sample_{i}.png',
            show=True
        )