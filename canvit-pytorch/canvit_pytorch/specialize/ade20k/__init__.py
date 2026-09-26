"""Linear ADE20K segmentation probes on frozen features (paper, Appendix D.3).

`python -m canvit_pytorch.specialize.ade20k canvas-probe` trains a probe on the canvas of a pretrained CanViT;
`dinov3-probe` trains one on DINOv3's patch features, the passive baseline.
"""
