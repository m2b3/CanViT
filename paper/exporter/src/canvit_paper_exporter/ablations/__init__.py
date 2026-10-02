from canvit_paper_exporter.ablations import in1k_clf, loss, recon, seg, static

DATASETS = [static.dataset, recon.dataset]
# Downstream evaluations of the ablation checkpoints, which the paper does not report; run by name.
DOWNSTREAM_DATASETS = [seg.dataset, in1k_clf.dataset]
FIGURES = [loss.figure]
