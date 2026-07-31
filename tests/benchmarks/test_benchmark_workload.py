import hashlib
from fractions import Fraction

import numpy as np
import pytest

from benchmarks.compare.einf_einops_einx_dynamic import (
    _build_case_specs,
    _compare_workloads_to_medium,
)
from benchmarks.harness import (
    BackendSpec,
    BenchSizes,
    DimensionMode,
    DynamicShapeWorkload,
    TensorGenerator,
    dynamic_sizes_for_scale,
)
from benchmarks.harness.generator import derive_coordinate_seed
from benchmarks.harness.runner import CASE_SEED_STRIDE


def test_dynamic_case_metadata_matches_configured_workloads() -> None:
    specs = _build_case_specs(sizes=dynamic_sizes_for_scale("large"))
    comparisons = _compare_workloads_to_medium(specs, scale="large")

    expected = {
        "rearrange_transpose_dynamic": {
            "dimensions": (
                ("b", "sampled", 24, 14, 33),
                ("n", "sampled", 384, 230, 537),
                ("d", "sampled", 128, 76, 179),
            ),
            "input_shapes": ((24, 384, 128),),
            "output_shapes": ((24, 128, 384),),
            "elements": (1_179_648, 1_179_648),
            "ratios": (
                ("b", Fraction(3, 2)),
                ("n", Fraction(2)),
                ("d", Fraction(4, 3)),
            ),
            "element_ratios": (Fraction(4), Fraction(4)),
        },
        "rearrange_flatten_hw_dynamic": {
            "dimensions": (
                ("b", "sampled", 24, 14, 33),
                ("h", "sampled", 48, 28, 67),
                ("w", "sampled", 32, 19, 44),
                ("d", "sampled", 128, 76, 179),
            ),
            "input_shapes": ((24, 48, 32, 128),),
            "output_shapes": ((24, 1536, 128),),
            "elements": (4_718_592, 4_718_592),
            "ratios": (
                ("b", Fraction(3, 2)),
                ("h", Fraction(3, 2)),
                ("w", Fraction(4, 3)),
                ("d", Fraction(4, 3)),
            ),
            "element_ratios": (Fraction(4), Fraction(4)),
        },
        "repeat_expand_axis_dynamic": {
            "dimensions": (
                ("b", "sampled", 24, 14, 33),
                ("d", "sampled", 128, 76, 179),
                ("r", "fixed", 24, 24, 24),
            ),
            "input_shapes": ((24, 128),),
            "output_shapes": ((24, 128, 24),),
            "elements": (3072, 73_728),
            "ratios": (
                ("b", Fraction(3, 2)),
                ("d", Fraction(4, 3)),
                ("r", Fraction(3, 2)),
            ),
            "element_ratios": (Fraction(2), Fraction(3)),
        },
        "reduce_sum_axes_dynamic": {
            "dimensions": (
                ("b", "sampled", 24, 14, 33),
                ("h", "sampled", 48, 28, 67),
                ("w", "sampled", 32, 19, 44),
                ("d", "sampled", 128, 76, 179),
            ),
            "input_shapes": ((24, 48, 32, 128),),
            "output_shapes": ((24, 128),),
            "elements": (4_718_592, 3072),
            "ratios": (
                ("b", Fraction(3, 2)),
                ("h", Fraction(3, 2)),
                ("w", Fraction(4, 3)),
                ("d", Fraction(4, 3)),
            ),
            "element_ratios": (Fraction(4), Fraction(2)),
        },
        "contract_matmul_dynamic": {
            "dimensions": (
                ("b", "sampled", 24, 14, 33),
                ("n", "sampled", 384, 230, 537),
                ("d", "sampled", 128, 76, 179),
                ("j", "sampled", 192, 115, 268),
            ),
            "input_shapes": ((24, 384, 128), (128, 192)),
            "output_shapes": ((24, 384, 192),),
            "elements": (1_204_224, 1_769_472),
            "ratios": (
                ("b", Fraction(3, 2)),
                ("n", Fraction(2)),
                ("d", Fraction(4, 3)),
                ("j", Fraction(3, 2)),
            ),
            "element_ratios": (Fraction(98, 25), Fraction(9, 2)),
        },
        "einop_contract_split_dynamic": {
            "dimensions": (
                ("b", "sampled", 24, 14, 33),
                ("h", "fixed", 48, 48, 48),
                ("w", "fixed", 32, 32, 32),
                ("r", "fixed", 24, 24, 24),
                ("n", "sampled", 384, 230, 537),
                ("d", "sampled", 128, 76, 179),
            ),
            "input_shapes": ((24, 1920, 384), (384, 128)),
            "output_shapes": ((24, 1152, 128), (24, 768, 128)),
            "elements": (17_743_872, 5_898_240),
            "ratios": (
                ("b", Fraction(3, 2)),
                ("h", Fraction(3, 2)),
                ("w", Fraction(4, 3)),
                ("r", Fraction(3, 2)),
                ("n", Fraction(2)),
                ("d", Fraction(4, 3)),
            ),
            "element_ratios": (Fraction(2888, 451), Fraction(30, 7)),
        },
    }

    assert {spec.case.name for spec in specs} == set(expected)
    for spec in specs:
        metadata = spec.workload_metadata
        comparison = comparisons[spec.case.name]
        expected_case = expected[spec.case.name]

        assert (
            tuple(
                (
                    dimension.name,
                    dimension.mode.value,
                    dimension.base,
                    dimension.minimum,
                    dimension.maximum,
                )
                for dimension in metadata.dimensions
            )
            == expected_case["dimensions"]
        )
        assert metadata.base_input_shapes == expected_case["input_shapes"]
        assert metadata.base_output_shapes == expected_case["output_shapes"]
        assert (
            metadata.base_input_elements,
            metadata.base_output_elements,
        ) == expected_case["elements"]
        assert comparison.dimension_ratios == expected_case["ratios"]
        assert (
            comparison.base_input_elements_ratio,
            comparison.base_output_elements_ratio,
        ) == expected_case["element_ratios"]


def test_dynamic_workload_output_shapes_match_case_references() -> None:
    sizes = BenchSizes(b=2, n=3, d=4, h=2, w=3, r=2, j=5)

    for spec in _build_case_specs(sizes=sizes):
        inputs = tuple(
            np.zeros(shape, dtype=np.float32)
            for shape in spec.workload_metadata.base_input_shapes
        )
        output = spec.case.reference(inputs)
        outputs = output if isinstance(output, tuple) else (output,)

        assert tuple(array.shape for array in outputs) == (
            spec.workload_metadata.base_output_shapes
        )


def test_dynamic_workload_preserves_seeded_batch_stream() -> None:
    expected_fingerprints = {
        "rearrange_transpose_dynamic": (
            (
                (20, 147, 93),
                "7f1432f6b2f855a8649364b6c33d1c07d6b16cd1db7602d0c9fe57f2fe33d1d7",
            ),
        ),
        "rearrange_flatten_hw_dynamic": (
            (
                (16, 32, 23, 115),
                "bdfc2750a5af4d699b3433d63dbb772917b479e294dfc0cd3102893f90897fd3",
            ),
        ),
        "repeat_expand_axis_dynamic": (
            (
                (19, 131),
                "b0e36f7ac7abdb4c80f54a1a1e9e3e1410685a57bd5a94d7f34b5752d0528745",
            ),
        ),
        "reduce_sum_axes_dynamic": (
            (
                (15, 22, 30, 76),
                "67f1d0b90421d48c0a515f3f3cc0dea351eaaa8b07caad881488bafb18a3dbaf",
            ),
        ),
        "contract_matmul_dynamic": (
            (
                (17, 161, 84),
                "c5e3a072dd10b073dfcfa9d8384eae4cdb1bb12f228fd44385442bc0ff6f9f0f",
            ),
            (
                (84, 150),
                "43e96c231811f720404cb38fb90df78efb883c9686e501e3c4d17570fd797b72",
            ),
        ),
        "einop_contract_split_dynamic": (
            (
                (14, 896, 154),
                "2222cc619e56fe22aa57a66708c5c5c95a9a00326390a88c411903c612806b6a",
            ),
            (
                (154, 134),
                "74c585258f478cfe330843f9057d28fa8f9c7d9142e27c05c3ef3374073e4144",
            ),
        ),
    }
    backend = BackendSpec(name="numpy")
    specs = _build_case_specs(sizes=dynamic_sizes_for_scale("medium"))

    for case_index, spec in enumerate(specs):
        generator = TensorGenerator.from_seed(
            backend=backend,
            seed=20260215 + case_index * CASE_SEED_STRIDE,
        )
        batch = spec.make_batch(generator)
        fingerprints = tuple(
            (
                array.shape,
                hashlib.sha256(np.asarray(array).tobytes()).hexdigest(),
            )
            for array in batch
        )

        assert fingerprints == expected_fingerprints[spec.case.name]


def test_coordinate_seed_derivation_separates_large_batch_streams() -> None:
    first = derive_coordinate_seed(
        seed=20260215,
        case_index=0,
        round_index=0,
        stream_index=7919,
    )
    second = derive_coordinate_seed(
        seed=20260215,
        case_index=0,
        round_index=1,
        stream_index=0,
    )

    assert first != second
    assert first == derive_coordinate_seed(
        seed=20260215,
        case_index=0,
        round_index=0,
        stream_index=7919,
    )


@pytest.mark.parametrize(
    ("component", "value"),
    (
        ("seed", -1),
        ("case_index", -1),
        ("round_index", -1),
        ("stream_index", -1),
    ),
)
def test_coordinate_seed_derivation_rejects_negative_components(
    component: str,
    value: int,
) -> None:
    components = {
        "seed": 20260215,
        "case_index": 0,
        "round_index": 0,
        "stream_index": 0,
    }
    components[component] = value

    with pytest.raises(ValueError, match="must be non-negative"):
        derive_coordinate_seed(**components)


def test_dynamic_workload_rejects_invalid_dimension_contracts() -> None:
    sizes = BenchSizes(b=10, n=3, d=1, h=1, w=1, r=1, j=1)
    backend = BackendSpec(name="numpy")

    with pytest.raises(ValueError, match="must be unique"):
        DynamicShapeWorkload(
            sampled_dimensions=("b", "b"),
            input_shapes=lambda dimensions: ((dimensions["b"],),),
            output_shapes=lambda dimensions: ((dimensions["b"],),),
        )

    unused = DynamicShapeWorkload(
        sampled_dimensions=("b", "n"),
        input_shapes=lambda dimensions: ((dimensions["b"],),),
        output_shapes=lambda dimensions: ((dimensions["b"],),),
    )
    with pytest.raises(ValueError, match="not used"):
        unused.metadata(sizes=sizes)

    conditional = DynamicShapeWorkload(
        sampled_dimensions=("b",),
        input_shapes=lambda dimensions: (
            (
                (dimensions["b"], dimensions["n"])
                if dimensions["b"] >= sizes.b
                else (dimensions["b"],)
            ),
        ),
        output_shapes=lambda dimensions: ((dimensions["b"],),),
    )
    metadata = conditional.metadata(sizes=sizes)
    generator = TensorGenerator.from_seed(backend=backend, seed=5)
    with pytest.raises(ValueError, match="must not depend on sampled values"):
        conditional.draw_input_shapes(
            sizes=sizes,
            generator=generator,
            metadata=metadata,
        )

    rank_conditional = DynamicShapeWorkload(
        sampled_dimensions=("b",),
        input_shapes=lambda dimensions: ((dimensions["b"],),),
        output_shapes=lambda dimensions: (
            (
                (dimensions["b"],)
                if dimensions["b"] >= sizes.b
                else (dimensions["b"], 1)
            ),
        ),
    )
    rank_metadata = rank_conditional.metadata(sizes=sizes)
    generator = TensorGenerator.from_seed(backend=backend, seed=5)
    with pytest.raises(ValueError, match="ranks must not depend"):
        rank_conditional.draw_input_shapes(
            sizes=sizes,
            generator=generator,
            metadata=rank_metadata,
        )

    generator = TensorGenerator.from_seed(backend=backend, seed=5)
    with pytest.raises(ValueError, match="does not match the configured sizes"):
        conditional.draw_input_shapes(
            sizes=BenchSizes(b=11, n=3, d=1, h=1, w=1, r=1, j=1),
            generator=generator,
            metadata=metadata,
        )


def test_benchmark_sizes_reject_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="dimension b must be positive"):
        BenchSizes(b=0, n=1, d=1, h=1, w=1, r=1, j=1)
    with pytest.raises(TypeError, match="dimension b must be an int"):
        BenchSizes(b=True, n=1, d=1, h=1, w=1, r=1, j=1)


def test_dimension_modes_are_partitioned() -> None:
    workload = DynamicShapeWorkload(
        sampled_dimensions=("b",),
        input_shapes=lambda dimensions: ((dimensions["b"], dimensions["d"]),),
        output_shapes=lambda dimensions: (
            (dimensions["b"], dimensions["d"], dimensions["r"]),
        ),
    )
    metadata = workload.metadata(sizes=BenchSizes(b=2, n=1, d=3, h=1, w=1, r=4, j=1))

    assert tuple(dimension.name for dimension in metadata.sampled_dimensions) == ("b",)
    assert all(
        dimension.mode is DimensionMode.SAMPLED
        for dimension in metadata.sampled_dimensions
    )
    assert tuple(dimension.name for dimension in metadata.fixed_dimensions) == (
        "d",
        "r",
    )
    assert all(
        dimension.mode is DimensionMode.FIXED for dimension in metadata.fixed_dimensions
    )
