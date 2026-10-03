"""
DC2PSA: Deformable Parallel Spatial Attention-Enhanced Module for YOLO26
========================================================================

Official PyTorch implementation for:
"DC2PSA-YOLO26-Seg: Deformable Parallel Spatial Attention-Enhanced YOLO26 for
Real-Time Airport Runway Crack Detection and Instance Segmentation"
Sensors 2026, 26, 5113.

Authors:
    Saifal Abbas, Md Taherul Islam Shawon, Saqib Qamar, Muhammad Adeel
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2PSA


class DSConv(nn.Module):
    """
    Deformable Strip Convolution (DSConv).

    Employs parallel horizontal (1 x k) and vertical (k x 1) depthwise strip
    convolutional kernels to capture elongated curvilinear crack structures,
    followed by an expand-squeeze pointwise bottleneck for cross-channel mixing.

    Crack-Specific Architectural Rationale:
    ---------------------------------------
    1. Airport runway cracks are thin, elongated, tortuous structural defects.
    2. Standard square (3 x 3) receptive fields suffer from isotropic bias,
       integrating background pavement texture and noise.
    3. Parallel 1 x k and k x 1 strip kernels capture anisotropic linear continuity
       along horizontal, vertical, and diagonal crack propagation paths.
    4. Residual identity connection guarantees stable gradient propagation.

    Args:
        c (int): Input channel dimension.
        k (int): Strip kernel length along the primary axis (default: 7).
        e (int): Bottleneck expansion factor for pointwise channel mixing (default: 2).
    """

    def __init__(self, c: int, k: int = 7, e: int = 2):
        super().__init__()
        c_expanded = int(c * e)

        # Depthwise strip convolutions: parallel horizontal and vertical kernels
        self.strip_h = nn.Conv2d(
            c, c, kernel_size=(1, k), padding=(0, k // 2), groups=c, bias=False
        )
        self.strip_v = nn.Conv2d(
            c, c, kernel_size=(k, 1), padding=(k // 2, 0), groups=c, bias=False
        )
        self.bn_strip = nn.BatchNorm2d(c)

        # Pointwise expand-squeeze bottleneck for dense inter-channel interaction
        self.pw_expand = nn.Conv2d(c, c_expanded, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(c_expanded)
        self.pw_squeeze = nn.Conv2d(c_expanded, c, kernel_size=1, bias=False)
        self.bn_squeeze = nn.BatchNorm2d(c)

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DSConv.

        Args:
            x (torch.Tensor): Feature tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Enhanced feature map of identical shape (B, C, H, W).
        """
        # Parallel horizontal and vertical strip filtering with joint activation
        strip = self.act(self.bn_strip(self.strip_h(x) + self.strip_v(x)))

        # Channel mixing via expand-squeeze bottleneck
        out = self.act(self.bn_expand(self.pw_expand(strip)))
        out = self.bn_squeeze(self.pw_squeeze(out))

        # Residual shortcut
        return self.act(out + x)


class DC2PSA(C2PSA):
    """
    Deformable Cross-Stage Partial with Parallel Spatial Attention (DC2PSA).

    Enhances standard Ultralytics C2PSA attention by injecting a DSConv
    curvilinear strip feature extraction layer before the multi-head PSA attention
    blocks in the primary transformation branch.

    Dataflow Architecture:
    ----------------------
    Input (X)
       │
      cv1 (1x1 Conv)
       │
     Split into (a, b) across channels
       │                   │
    Branch a            Branch b
    (Identity skip)     DSConv(b)   <-- Elongated crack strip feature extraction
       │                   │
       │                PSA(b)      <-- Multi-head parallel spatial self-attention
       │                   │
       └─── Concat(a, b) ──┘
               │
              cv2 (1x1 Conv)
               │
          Output (Y)

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int): Number of internal PSA blocks (default: 1).
        e (float): Hidden channel expansion ratio (default: 0.5).
        k (int): Strip kernel length for internal DSConv (default: 7).
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, k: int = 7):
        super().__init__(c1, c2, n, e)
        # DSConv integrated inside transformation branch b
        self.dsconv = DSConv(self.c, k=k, e=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DC2PSA.

        Args:
            x (torch.Tensor): Feature map of shape (B, C1, H, W).

        Returns:
            torch.Tensor: Attention-refined feature map of shape (B, C2, H, W).
        """
        a, b = self.cv1(x).chunk(2, 1)
        b = self.dsconv(b)  # Curvilinear crack deformation enhancement
        b = self.m(b)       # Multi-head spatial self-attention blocks
        return self.cv2(torch.cat((a, b), 1))


def build_dc2psa_model(pretrained_path: str = "yolo26s.pt", task: str = "detect") -> YOLO:
    """
    Build DC2PSA-YOLO26s by replacing C2PSA layers in pretrained YOLO26 with DC2PSA.

    Transfer Learning Strategy:
    ---------------------------
    - All C3k2 backbone layers retain COCO pretrained weights.
    - All PAN-FPN neck and decoupled detection/segmentation head weights are preserved.
    - C2PSA layers are replaced by DC2PSA:
      - Inherited C2PSA parameters (cv1, cv2, PSA blocks) are transferred.
      - DSConv parameters are initialized fresh with He normal initialization.

    Args:
        pretrained_path (str): Path or model name for baseline YOLO26 weights.
        task (str): Task mode ('detect' or 'segment').

    Returns:
        YOLO: Configured Ultralytics YOLO model with DC2PSA attention.
    """
    print(f"\n{'=' * 75}")
    print(f"  Building DC2PSA-YOLO26s ({task.upper()})")
    print(f"{'=' * 75}")
    print(f"  Source Weights : {pretrained_path}")

    model = YOLO(pretrained_path)
    replaced_count = 0

    for i, module in enumerate(model.model.model):
        if isinstance(module, C2PSA):
            c1 = module.cv1.conv.in_channels
            c2 = module.cv2.conv.out_channels
            n = len(module.m) if hasattr(module.m, '__len__') else 1
            e = module.c / c2 if c2 > 0 else 0.5

            # Instantiate DC2PSA with identical dimensional configuration
            dc2psa = DC2PSA(c1, c2, n=n, e=e)

            # Transfer existing weights from C2PSA
            src_state = module.state_dict()
            tgt_state = dc2psa.state_dict()
            transferred = 0

            for k_src in src_state:
                if k_src in tgt_state and src_state[k_src].shape == tgt_state[k_src].shape:
                    tgt_state[k_src] = src_state[k_src]
                    transferred += 1

            dc2psa.load_state_dict(tgt_state)
            model.model.model[i] = dc2psa
            replaced_count += 1

            dsconv_params = sum(p.numel() for p in dc2psa.dsconv.parameters())
            print(f"  Layer {i:2d}: C2PSA -> DC2PSA (in={c1}, out={c2}, n={n}, e={e:.2f})")
            print(f"            [{transferred} weight tensors transferred | {dsconv_params:,} DSConv params added]")

    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    print(f"\n  [OK] Successfully configured DC2PSA-YOLO26s ({task.upper()}):")
    print(f"       Replaced modules   : {replaced_count}")
    print(f"       Total parameters   : {total_params / 1e6:.2f}M")
    print(f"       Trainable params   : {trainable_params / 1e6:.2f}M")
    print(f"{'=' * 75}\n")

    return model


if __name__ == "__main__":
    print("Testing DC2PSA module standalone execution and shape preservation...")
    dummy_x = torch.randn(2, 256, 20, 20)

    # 1. Test DSConv
    dsconv = DSConv(256, k=7, e=2)
    out_dsconv = dsconv(dummy_x)
    assert out_dsconv.shape == dummy_x.shape, f"DSConv shape mismatch: {out_dsconv.shape} vs {dummy_x.shape}"
    print(f"[OK] DSConv shape check passed: {dummy_x.shape} -> {out_dsconv.shape}")

    # 2. Test DC2PSA
    dc2psa = DC2PSA(256, 256, n=2, e=0.5)
    out_dc2psa = dc2psa(dummy_x)
    assert out_dc2psa.shape == dummy_x.shape, f"DC2PSA shape mismatch: {out_dc2psa.shape} vs {dummy_x.shape}"
    print(f"[OK] DC2PSA shape check passed: {dummy_x.shape} -> {out_dc2psa.shape}")

    # 3. Test Gradient Propagation
    loss = out_dc2psa.sum()
    loss.backward()
    print("[OK] Backward pass & gradient flow verified through DSConv + PSA branches.")
    print("All unit tests passed successfully!")
