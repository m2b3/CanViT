"""Teacher features computed once and read by every pretraining run (paper, Section 5.1).

The frozen teacher's features of every training scene are exported to shards
of 4,096 images; pretraining then reads them instead of running the teacher.

    python -m canvit_pytorch.pretrain.features.index    list the images, shuffled
    python -m canvit_pytorch.pretrain.features.export   compute and write shards
    python -m canvit_pytorch.pretrain.features.check    verify a complete shard tree
"""
