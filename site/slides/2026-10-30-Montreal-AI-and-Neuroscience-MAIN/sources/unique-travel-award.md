# UNIQUE travel award application: CanViT

[Written by Yohaï-Eliel Berreby for a UNIQUE (Unifying Neuroscience and Artificial Intelligence – Québec) travel
award; supplied by him on 2026-10-01 as the framing that worked with a neuro-AI audience, "subtle nuances and
formulations" included.] Verbatim.

## Brief description of the involvement of neuroscience AND AI in the project (max. 700 characters)

Leading vision foundation models used to model brain function lack the active sensing and top-down recurrent
feedback of biological vision. This limits their value as models of human vision, which fundamentally relies on eye
movements. With CanViT, we propose a biologically inspired, task- and policy-agnostic AI model that pairs
retinotopic glimpse processing with spatiotopic memory. CanViT sets a new state of the art in active computer vision
by a wide margin, with unprecedented efficiency on active semantic segmentation. With open code, open weights, and
fast training, we hope CanViT will serve the neuroscience and AI communities alike as the first of many Active-Vision
Foundation Models.

## Summary (max. 2500 characters)

Human vision is fundamentally active: we shift our gaze multiple times per second, orienting our high-resolution
fovea toward regions of interest, and integrate information from multiple viewpoints into a coherent understanding
of our surroundings' spatial and semantic structure. Yet, the Vision Foundation Models that increasingly serve as
models of visual cognition lack essential active-vision components, such as sequential localized glimpses or memory.
Here, we propose the Canvas Vision Transformer (CanViT), the first task- and policy-agnostic Active-Vision Foundation
Model (AVFM). CanViT decouples sensory processing from working memory by pairing a retinotopic Vision Transformer
backbone with a spatiotopic latent workspace, the canvas. The canvas can function as a cognitive map of the scene,
providing a stable spatial reference frame and top-down recurrent feedback to the backbone's visual processing.
Efficient interaction with this high-capacity working memory is supported by Canvas Attention, a novel asymmetric
cross-attention mechanism. We disentangle "learning to see through glimpses" from "learning where to look" with a
policy-agnostic pretraining scheme, sampling 1 billion random glimpses from 13.2 million ImageNet-21K scenes. We
introduce a passive-to-active dense latent distillation objective, training CanViT to approximate scene-wide
spatio-semantic embeddings produced by a high-resolution DINOv3 teacher. This rich learning signal allows CanViT to
quickly inherit much of its teacher's world knowledge, while developing the ability to iteratively refine its
understanding of a scene as it ingests glimpses at arbitrary locations and zoom levels. CanViT pretraining completes
in just 166 hours on a single H100 GPU. Our model achieves a high degree of visuospatial intelligence, setting a new
active-vision state of the art on ADE20K semantic segmentation: with frozen weights, CanViT-B achieves 38.5% mIoU in
a single low-resolution glimpse, outperforming the best active model's 27.6% with 20x fewer FLOPs. Given additional
glimpses, it reaches 45.9% mIoU. On ImageNet-1K, CanViT-B also sets a new active-vision state of the art, with 84.5%
top-1 accuracy. CanViT generalizes to longer rollouts, larger scenes, and new viewing policies. Our work narrows the
wide gap between passive and active computer vision, demonstrates the potential of AVFMs as a new research axis, and
constitutes a major step toward action-aware neuro-AI research on vision.
