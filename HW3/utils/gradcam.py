import torch
import logging


logger = logging.getLogger("HW3")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Forward hook — stores the feature maps produced by target_layer."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Backward hook — stores the gradients w.r.t. target_layer output."""
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx=None):
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # [H,W] → [1,1,H,W]

        elif x.dim() == 3:
            x = x.unsqueeze(0)

        device = next(self.model.parameters()).device
        x = x.to(device)
        x.requires_grad = True
        with torch.set_grad_enabled(True):
            self.model.zero_grad()
            logits = self.model(x)

            if class_idx is None:
                class_idx = logits.argmax(dim=1)

            loss = logits[0, class_idx]
            loss.backward()

        # Compute weights: global average pool of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # Weighted combination
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze()

        # Min-max normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        """Deregister both hooks. Call when done to avoid memory leaks."""
        self.fwd_handle.remove()
        self.bwd_handle.remove()
