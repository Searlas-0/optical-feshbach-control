import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ofc.config import (
    InitializationQuery,
    ResolvedConfig,
    RuntimeConfig,
    batch_key,
    load_config,
    make_document,
    write_config,
)


MAKER_PATH = Path(__file__).resolve().parents[1] / "run_config" / "make_config.py"
MAKER_SPEC = importlib.util.spec_from_file_location("ofc_make_config", MAKER_PATH)
MAKER = importlib.util.module_from_spec(MAKER_SPEC)
assert MAKER_SPEC.loader is not None
MAKER_SPEC.loader.exec_module(MAKER)
default_parameters = MAKER.default_parameters
default_runtime = MAKER.default_runtime
make_config_file = MAKER.make_config


def test_cartesian_sweeps_batch_compatible_parameters_and_order_schedule_before_n():
    document = make_document(
        name="matrix",
        parameters={
            "N": [4, 8],
            "schedule": [[(1, 1.0)], [(2, 1.0), (1, 0.5)]],
            "u_max": [10.0, 20.0, 30.0],
            "smoothness": [0.0, 1e-3],
        },
    )

    assert len(document.scalar_cases()) == 24
    batches = document.batches()
    assert [(batch.N, len(batch.schedule)) for batch in batches] == [
        (4, 1),
        (4, 2),
        (8, 1),
        (8, 2),
    ]
    assert all(len(batch.cases) == 6 for batch in batches)
    assert len({batch.batch_id for batch in batches}) == 4
    assert all(batch.seed == (document.config_id + batch.batch_id) % (2**32 - 1) for batch in batches)


def test_written_config_round_trip_retains_random_ids(tmp_path):
    document = make_document(
        name="round_trip",
        parameters={"N": 4},
        query={
            "where": {"config_name": "source", "best_score": "0.1:"},
            "limit": 3,
            "order_by": "best_score",
            "descending": True,
            "control_kind": "best",
        },
    )
    path = write_config(document, tmp_path / "round_trip.yaml")
    restored = load_config(path)

    assert restored.config_id == document.config_id
    assert restored.batch_ids == document.batch_ids
    assert restored.batches()[0].seed == document.batches()[0].seed
    assert restored.config_file == "round_trip.yaml"
    assert restored.query == document.query
    with pytest.raises(FileExistsError, match="new config name"):
        write_config(document, path)


def test_initialization_query_validation_and_defaults():
    query = InitializationQuery(where={"config_name": "source"})

    assert query.limit is None
    assert query.order_by == "best_score"
    assert query.descending is True
    assert query.control_kind == "best"

    with pytest.raises(ValueError, match="non-empty mapping"):
        InitializationQuery(where={})
    with pytest.raises(ValueError, match="positive integer"):
        InitializationQuery(where={"run_id": 1}, limit=0)
    with pytest.raises(ValueError, match="initial, best, or final"):
        InitializationQuery(where={"run_id": 1}, control_kind="optimal")


def test_config_maker_uses_fresh_defaults_when_called_without_overrides(tmp_path):
    first = load_config(make_config_file(output_dir=tmp_path))
    second = load_config(make_config_file(output_dir=tmp_path))

    assert first.name.startswith("default_")
    assert second.name.startswith("default_")
    assert first.name != second.name
    assert first.scalar_cases() == (ResolvedConfig(**default_parameters()),)
    assert first.runtime == RuntimeConfig(**default_runtime())
    assert first.config_id != second.config_id


def test_config_maker_applies_only_explicit_overrides(tmp_path):
    path = make_config_file(
        name="small_sweep",
        description="test",
        parameters={"N": 4, "u_max": [10.0, 20.0]},
        runtime={"initialisations": 2, "device": "cpu"},
        output_dir=tmp_path,
    )
    document = load_config(path)

    assert document.parameters["N"] == 4
    assert document.parameters["u_max"] == [10.0, 20.0]
    assert document.parameters["v_max"] == default_parameters()["v_max"]
    assert document.runtime.initialisations == 2
    assert document.runtime.fourier_num_modes == default_runtime()["fourier_num_modes"]


def test_config_maker_accepts_numpy_arrays_as_sweeps(tmp_path):
    u_max = np.logspace(-1, 2, 13)
    path = make_config_file(
        name="numpy_u_max_sweep",
        parameters={"N": [100, 200], "u_max": u_max},
        runtime={"concurrent_workers": np.int64(2)},
        output_dir=tmp_path,
    )
    document = load_config(path)

    assert len(document.scalar_cases()) == 26
    assert np.allclose(
        sorted({case.u_max for case in document.scalar_cases()}),
        u_max,
    )
    assert document.runtime.concurrent_workers == 2
    assert isinstance(document.parameters["u_max"], list)


def test_compile_shape_fields_cannot_be_swept():
    with pytest.raises(ValueError, match="cannot be swept"):
        make_document(name="bad", parameters={"block_size": [10, 20]})


def test_time_sweep_creates_stable_independent_batches_for_slurm_arrays():
    document = make_document(
        name="time_shards",
        parameters={"N": [100, 200], "t_interval": [0.1, 1.0, 10.0]},
    )

    assert [
        (batch.batch_index, batch.N, batch.t_interval)
        for batch in document.batches()
    ] == [
        (0, 100, 0.1),
        (1, 100, 1.0),
        (2, 100, 10.0),
        (3, 200, 0.1),
        (4, 200, 1.0),
        (5, 200, 10.0),
    ]


def test_r_bg_defaults_to_one_and_sweeps_signed_nonzero_values():
    assert ResolvedConfig().r_bg == 1.0
    document = make_document(
        name="background_ratios",
        parameters={"N": 4, "r_bg": [-2.5, 0.75]},
    )

    assert [case.r_bg for case in document.scalar_cases()] == [-2.5, 0.75]
    assert len(document.batches()) == 1
    assert len(document.batches()[0].cases) == 2


@pytest.mark.parametrize(
    "value", [0, 0.0, True, float("inf"), float("-inf"), float("nan"), "1"]
)
def test_r_bg_rejects_zero_nonfinite_and_nonnumeric_values(value):
    with pytest.raises(ValueError, match="finite non-zero"):
        ResolvedConfig(r_bg=value)


def test_optimizer_settings_are_separate_and_sharpness_overrides_are_resolved():
    adam = ResolvedConfig(
        optimizer="ADAM",
        adam_learning_rate=2e-3,
        adam_beta1=0.8,
        sharpness=0.25,
        u_sharp=0.5,
    )
    lbfgs = ResolvedConfig(
        optimizer="lbfgs",
        schedule=((4, 1.0),),
        lbfgs_history_size=7,
        lbfgs_max_linesearch_steps=12,
        lbfgs_tolerance=2e-7,
        sharpness=0.25,
        v_sharp=0.75,
    )

    assert adam.optimizer == "adam"
    assert adam.adam_learning_rate == 2e-3
    assert adam.adam_beta1 == 0.8
    assert adam.effective_u_sharp == 0.5
    assert adam.effective_v_sharp == 0.25
    assert lbfgs.optimizer == "lbfgs"
    assert lbfgs.lbfgs_history_size == 7
    assert lbfgs.lbfgs_max_linesearch_steps == 12
    assert lbfgs.lbfgs_tolerance == 2e-7
    assert lbfgs.effective_u_sharp == 0.25
    assert lbfgs.effective_v_sharp == 0.75


def test_zero_and_nonzero_sharpness_compile_as_separate_batches():
    document = make_document(
        name="sharpness_profiles",
        parameters={"N": 4, "sharpness": [0.0, 1e-4]},
    )

    assert len(document.batches()) == 2
    assert [batch.cases[0].sharpness for batch in document.batches()] == [0.0, 1e-4]
    assert document.batches()[0].key == document.batches()[1].key
    assert document.batches()[0].batch_id == document.batches()[1].batch_id


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"optimizer": "sgd"}, "optimizer"),
        (
            {"optimizer": "lbfgs", "schedule": ((2, 0.5),)},
            "multipliers",
        ),
        ({"sharpness": -1e-3}, "non-negative"),
        ({"lbfgs_history_size": 0}, "positive integer"),
        ({"lbfgs_tolerance": 0.0}, "positive number"),
    ],
)
def test_invalid_optimizer_and_sharpness_settings_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        ResolvedConfig(**overrides)


def test_batch_key_is_canonical():
    assert json.loads(batch_key(4, [(2, 1)])) == {
        "N": 4,
        "schedule": [[2, 1.0]],
        "t_interval": 1.0,
    }
