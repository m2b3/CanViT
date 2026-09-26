"""Per-mask IoU on ADE20K: the data behind the paper's Figure 5A-B (methodology in Appendix D.5).

Each (validation image, class) pair gets its intersection, union and ground-truth area in pixels,
so the paper can plot IoU against mask area. The DINOv3 stage covers the passive teacher at
128 px; the CanViT stage covers canvas grids 8² to 64² after every EG-C2F glimpse. Both write a
parquet table with one row per pair and a JSON file describing the runs.
"""
