import torch


class Saturation:
    """
    Adjust color saturation of an image.
    Implementation taken from
    https://kornia.readthedocs.io/en/latest/_modules/kornia/color/adjust.html#adjust_saturation
    and modified.
    """
    def __init__(self, factor):
        # Initialize
        self.saturation_factor = torch.tensor([factor])
        self.saturation_factor.requires_grad_(True)

    def get_tensor(self, _img):
        # construct transformation within tensors
        h, s, v = torch.chunk(_img, chunks=3, dim=-3)
        s_out: torch.Tensor = torch.clamp(s * self.saturation_factor, min=0, max=1)
        out: torch.Tensor = torch.cat([h, s_out, v], dim=-3).requires_grad_(True)
        return out

    def get_gradient(self):
        # get gradient with respect to the saturation factor
        return self.saturation_factor.grad.item()

    def retain_grad(self):
        # retain graph when factor is not a leaf node
        self.saturation_factor.retain_grad()


class Contrast:
    """
    Adjust contrast of an image.
    Implementation taken from
    https://kornia.readthedocs.io/en/latest/_modules/kornia/color/adjust.html#adjust_contrast
    and modified.
    """
    def __init__(self, factor):
        # Initialize
        self.contrast_factor = torch.tensor([factor])
        self.contrast_factor.requires_grad_(True)

    def get_tensor(self, _img):
        # construct transformation within tensors
        for _ in _img.shape[1:]:
            self.contrast_factor = torch.unsqueeze(self.contrast_factor, dim=-1).requires_grad_(True)
        x_adjust: torch.Tensor = _img * self.contrast_factor.requires_grad_(True)
        out: torch.Tensor = torch.clamp(x_adjust, 0.0, 1.0)
        return out

    def get_gradient(self):
        # get gradient with respect to the contrast factor
        return self.contrast_factor.grad.item()

    def retain_grad(self):
        # retain graph when factor is not a leaf node
        self.contrast_factor.retain_grad()


class Brightness:
    """
    Adjust brightness of an image.
    Implementation taken from
    https://kornia.readthedocs.io/en/latest/_modules/kornia/color/adjust.html#adjust_brightness
    and modified.
    """

    def __init__(self, factor):
        # Initialize
        self.brightness_factor = torch.tensor([factor])
        self.brightness_factor.requires_grad_(True)

    def get_tensor(self, _img):
        # construct transformation within tensors
        for _ in _img.shape[1:]:
            self.brightness_factor = torch.unsqueeze(self.brightness_factor, dim=-1).requires_grad_(True)
        x_adjust: torch.Tensor = _img + self.brightness_factor
        out: torch.Tensor = torch.clamp(x_adjust, 0.0, 1.0).requires_grad_(True)
        return out

    def get_gradient(self):
        # get gradient with respect to the brightness factor
        return self.brightness_factor.grad.item()

    def retain_grad(self):
        # retain graph when factor is not a leaf node
        self.brightness_factor.retain_grad()
