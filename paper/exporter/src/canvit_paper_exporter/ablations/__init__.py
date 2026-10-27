from canvit_paper_exporter.ablations import in1k_clf, loss, recon, seg, static, vpe

DATASETS = [static.dataset, recon.dataset, seg.dataset, in1k_clf.dataset, vpe.dataset]
FIGURES = [loss.figure]
