import pytest


def test_mlx_permanent_buffers_stay_out_of_trainable_parameters(backend, tiny_config):
    if backend.name != "mlx":
        pytest.skip("MLX permanent-buffer behavior")

    from mlx.utils import tree_flatten

    def names(tree):
        return {name for name, _ in tree_flatten(tree)}

    models = (
        (
            backend.make_canvit(tiny_config),
            {"vpe.frequencies"},
        ),
        (
            backend.make_pretraining(tiny_config, teacher_dim=4, teacher_patch_grid=2),
            {
                "canvit.vpe.frequencies",
                "teacher_patch_standardizer.mean",
                "teacher_patch_standardizer.var",
                "teacher_patch_standardizer.fitted",
                "teacher_cls_standardizer.mean",
                "teacher_cls_standardizer.var",
                "teacher_cls_standardizer.fitted",
            },
        ),
        (
            backend.module.SegmentationProbe(
                embed_dim=tiny_config.canvas_dim,
                num_classes=3,
                dropout=0.1,
                use_ln=True,
            ),
            {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"},
        ),
    )
    for model, permanent in models:
        all_names = names(model.parameters())
        before = names(model.trainable_parameters())
        model.freeze()
        model.unfreeze()
        after = names(model.trainable_parameters())
        assert permanent <= all_names
        assert not permanent & before
        assert not permanent & after
        assert before == after
