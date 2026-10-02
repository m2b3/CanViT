import numpy as np
import torch
from canvit_core import CanViTConfig
from canvit_pytorch import CanViTForImageClassification, CanViTForPretraining

from tests.backends.support import Backend, canvit_values, normalized_l2, relative_l2


def test_pretraining_readouts_and_classifier_logits_match_pytorch(backend: Backend, tiny_config: CanViTConfig):
    torch.manual_seed(11)
    torch_pretraining = CanViTForPretraining(
        canvit_config=tiny_config, teacher_dim=12, teacher_patch_grid=4,
    ).eval()
    native_pretraining = backend.make_pretraining(tiny_config, teacher_dim=12, teacher_patch_grid=4)
    backend.load_reference_weights(native_pretraining, torch_pretraining)
    torch_classifier = CanViTForImageClassification(canvit_config=tiny_config, n_classes=7).eval()
    native_classifier = backend.make_classifier(tiny_config, n_classes=7)
    backend.load_reference_weights(native_classifier, torch_classifier)

    torch_glimpse, torch_viewpoint, native_glimpse, native_viewpoint = backend.inputs()
    torch_pre_state = torch_pretraining.init_state(batch_size=2, canvas_grid_size=4)
    native_pre_state = native_pretraining.init_state(batch_size=2, canvas_grid_size=4)
    torch_cls_state = torch_classifier.init_state(batch_size=2, canvas_grid_size=4)
    native_cls_state = native_classifier.init_state(batch_size=2, canvas_grid_size=4)
    with torch.inference_mode():
        torch_pre_output = torch_pretraining(
            glimpse=torch_glimpse, state=torch_pre_state, viewpoint=torch_viewpoint,
        )
        torch_pre_values = canvit_values(torch_pre_output, backend)
        torch_pre_values["teacher_patches"] = torch_pretraining.predict_teacher_patches(
            torch_pre_output.state.canvas,
        ).numpy()
        torch_pre_values["teacher_cls"] = torch_pretraining.predict_teacher_cls(
            torch_pre_output.state.recurrent_cls,
        ).numpy()
        torch_logits, torch_cls_state = torch_classifier(
            glimpse=torch_glimpse, state=torch_cls_state, viewpoint=torch_viewpoint,
        )
    native_pre_output = native_pretraining(
        glimpse=native_glimpse, state=native_pre_state, viewpoint=native_viewpoint,
    )
    native_pre_values = canvit_values(native_pre_output, backend)
    native_pre_values["teacher_patches"] = backend.to_numpy(
        native_pretraining.predict_teacher_patches(native_pre_output.state.canvas),
    )
    native_pre_values["teacher_cls"] = backend.to_numpy(
        native_pretraining.predict_teacher_cls(native_pre_output.state.recurrent_cls),
    )
    for name, reference in torch_pre_values.items():
        assert relative_l2(reference, native_pre_values[name]) < 2e-3, name
        assert normalized_l2(reference, native_pre_values[name]) < 2e-3, name
    native_logits, _ = native_classifier(
        glimpse=native_glimpse, state=native_cls_state, viewpoint=native_viewpoint,
    )
    np.testing.assert_allclose(backend.to_numpy(native_logits), torch_logits.numpy(), atol=2e-4, rtol=2e-3)
