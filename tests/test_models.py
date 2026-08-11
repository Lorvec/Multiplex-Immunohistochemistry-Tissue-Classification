import pytest


torch = pytest.importorskip("torch")

from mihic_repro.models import ModernMIHICNet, count_parameters  # noqa: E402


@pytest.mark.parametrize(
    ("variant", "expected_parameters"),
    [
        ("nano", 299_843),
        ("tiny", 819_427),
        ("small", 2_528_199),
    ],
)
def test_frozen_family_parameter_counts_and_output_shape(
    variant, expected_parameters
):
    model = ModernMIHICNet(
        num_classes=7,
        variant=variant,
        use_attention=False,
        multi_scale=True,
    ).eval()
    assert count_parameters(model) == expected_parameters
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 128, 128))
    assert tuple(output.shape) == (1, 7)
