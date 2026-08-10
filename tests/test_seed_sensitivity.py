from ofc.seed_sensitivity import (
    STAGES,
    fixed_cap_seed_sensitivity_documents,
    fixed_endpoint_seed_sensitivity_documents,
    fixed_seed_stage_name,
    strict_refinement_document,
)


CAP_SETTINGS = {
    1280: {
        "adam_learning_rate": 0.15,
        "adam_beta1": 0.95,
        "adam_beta2": 0.999,
        "smoothness": 2.5e-7,
        "sharpness": 2.5e-8,
    },
    160: {
        "adam_learning_rate": 0.05,
        "adam_beta1": 0.95,
        "adam_beta2": 0.99,
        "smoothness": 1.25e-7,
        "sharpness": 1.25e-8,
    },
    40: {
        "adam_learning_rate": 0.15,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "smoothness": 1.25e-7,
        "sharpness": 5e-8,
    },
}


def documents():
    return fixed_cap_seed_sensitivity_documents(
        cap_settings=CAP_SETTINGS,
        resolutions=(100, 200, 300, 400, 500),
        database="results/bar_seed_sensitivity.sqlite3",
        batch_sizes={100: 75, 200: 40, 300: 28, 400: 20, 500: 16},
    )


def test_fixed_cap_seed_sensitivity_builds_four_ordered_stages_per_case():
    generated = documents()

    assert len(generated) == 5 * 3 * 4
    expected = []
    for N in (100, 200, 300, 400, 500):
        for cap in (1280, 160, 40):
            expected.extend(fixed_seed_stage_name(N, cap, stage) for stage in STAGES)
    assert [document.name for document in generated] == expected


def test_stage_names_track_nondefault_seed_count_for_quick_screens():
    generated = fixed_cap_seed_sensitivity_documents(
        cap_settings={320: CAP_SETTINGS[160]},
        resolutions=(100,),
        database="results/bar_u320_crossover_screen.sqlite3",
        exploration_initialisations=250,
        exploration_schedule=((2_500, 1.5), (7_500, 0.75)),
        all_polish_steps=500,
        batch_sizes={100: 25},
        loose_max_elapsed_seconds=30 * 60,
    )

    assert [document.name for document in generated] == [
        fixed_seed_stage_name(100, 320, stage, 250) for stage in STAGES
    ]
    assert generated[0].runtime.initialisations == 250
    assert generated[0].scalar_cases()[0].schedule == (
        (2_500, 1.5),
        (7_500, 0.75),
    )
    assert generated[1].scalar_cases()[0].schedule == ((500, 1.0),)
    assert generated[2].runtime.max_elapsed_seconds == 30 * 60


def test_endpoint_study_interleaves_low_and_high_pairs_before_next_resolution():
    endpoint_settings = {
        160: {
            "low": {
                **CAP_SETTINGS[160],
                "smoothness": 3.952847075210474e-9,
                "sharpness": 3.952847075210474e-10,
            },
            "high": {
                **CAP_SETTINGS[160],
                "smoothness": 3.9528470752104736e-6,
                "sharpness": 3.952847075210474e-7,
            },
        }
    }
    generated = fixed_endpoint_seed_sensitivity_documents(
        cap_endpoint_settings=endpoint_settings,
        resolutions=(100, 200),
        database="results/endpoints.sqlite3",
        batch_sizes={100: 25, 200: 20},
    )

    assert len(generated) == 16
    assert [document.name for document in generated[0:9:4]] == [
        fixed_seed_stage_name(
            100, 160, "exploration", parameter_label="low_regularization"
        ),
        fixed_seed_stage_name(
            100, 160, "exploration", parameter_label="high_regularization"
        ),
        fixed_seed_stage_name(
            200, 160, "exploration", parameter_label="low_regularization"
        ),
    ]
    low_case = generated[0].scalar_cases()[0]
    high_case = generated[4].scalar_cases()[0]
    assert low_case.smoothness == 3.952847075210474e-9
    assert low_case.sharpness == 3.952847075210474e-10
    assert high_case.smoothness == 3.9528470752104736e-6
    assert high_case.sharpness == 3.952847075210474e-7

    bar_only = fixed_endpoint_seed_sensitivity_documents(
        cap_endpoint_settings=endpoint_settings,
        resolutions=(100,),
        database="results/endpoints.sqlite3",
        batch_sizes={100: 25},
        include_strict=False,
        parameter_label_suffix="_bar_v2",
    )
    assert len(bar_only) == 6
    assert all("top1_strict" not in document.name for document in bar_only)
    assert all("regularization_bar_v2" in document.name for document in bar_only)


def test_all_seed_loose_refinement_uses_memory_safe_batches_and_per_batch_timeout():
    generated = fixed_cap_seed_sensitivity_documents(
        cap_settings={320: CAP_SETTINGS[160]},
        resolutions=(100, 500),
        database="results/all_seed_loose.sqlite3",
        exploration_initialisations=1_000,
        batch_sizes={100: 75, 500: 16},
        loose_batch_sizes={100: 40, 500: 10},
        top_count=1_000,
        loose_max_elapsed_seconds=45 * 60,
    )

    for offset, expected_batch_size in ((0, 40), (4, 10)):
        top_loose = generated[offset + 2]
        assert "top1000_loose" in top_loose.name
        assert top_loose.query.limit == 1_000
        assert top_loose.runtime.max_initialisations_per_batch == expected_batch_size
        assert top_loose.runtime.max_batch_elapsed_seconds == 45 * 60
        assert top_loose.runtime.max_elapsed_seconds is None


def test_exploration_is_broad_gpu_batched_and_never_exceeds_twenty_thousand_steps():
    generated = documents()

    for document in generated[0::4]:
        N = document.scalar_cases()[0].N
        case = document.scalar_cases()[0]
        assert case.optimizer == "adam"
        assert sum(steps for steps, _ in case.schedule) == 20_000
        assert max(multiplier for _, multiplier in case.schedule) > 1.0
        assert case.J_tol is None
        assert case.u_tol is None
        assert case.v_tol is None
        assert case.projected_gradient_tol is None
        assert document.runtime.initialisations == 1_000
        assert document.runtime.fourier_num_modes == 6
        assert document.runtime.fourier_rms_amplitude == 0.8
        assert document.runtime.fourier_intensity_fraction == 0.5
        assert document.runtime.device == "gpu"
        assert document.runtime.use_jit is True
        assert document.runtime.max_initialisations_per_batch == {
            100: 75,
            200: 40,
            300: 28,
            400: 20,
            500: 16,
        }[N]


def test_every_handoff_uses_fresh_optimizer_state_and_exact_source_config_id():
    generated = documents()

    for offset in range(0, len(generated), 4):
        exploration, all_polish, top_loose, top_strict = generated[offset : offset + 4]
        assert all_polish.query.where["config_id"] == exploration.config_id
        assert top_loose.query.where["config_id"] == all_polish.config_id
        assert top_strict.query.where["config_id"] == top_loose.config_id
        assert all_polish.query.order_by == "best_objective"
        assert top_loose.query.order_by == "best_objective"
        assert top_strict.query.order_by == "best_objective"
        assert all_polish.query.limit is None
        assert top_loose.query.limit == 20
        assert top_strict.query.limit == 1
        assert all(
            document.query.resume_optimizer is False
            for document in (all_polish, top_loose, top_strict)
        )


def test_polish_loose_and_strict_stages_have_the_requested_roles_and_tolerances():
    generated = documents()

    for offset in range(0, len(generated), 4):
        _, all_polish, top_loose, top_strict = generated[offset : offset + 4]
        polish_case = all_polish.scalar_cases()[0]
        loose_case = top_loose.scalar_cases()[0]
        strict_case = top_strict.scalar_cases()[0]

        assert polish_case.optimizer == "peak_refinement"
        assert polish_case.schedule == ((2_000, 1.0),)
        assert polish_case.J_tol is None
        assert polish_case.projected_gradient_tol is None
        assert all_polish.runtime.repeat_schedule_until_stable is False

        assert loose_case.optimizer == "lbfgs"
        assert loose_case.J_tol == 1e-5
        assert loose_case.u_tol == loose_case.v_tol == 1e-4
        assert loose_case.projected_gradient_tol == 1e-4
        assert top_loose.runtime.repeat_schedule_until_stable is True
        assert top_loose.runtime.max_elapsed_seconds == 2 * 60 * 60

        assert strict_case.optimizer == "peak_refinement"
        assert strict_case.J_tol == 1e-6
        assert strict_case.u_tol == strict_case.v_tol == 1e-5
        assert strict_case.projected_gradient_tol == 1e-5
        assert top_strict.runtime.repeat_schedule_until_stable is True
        assert top_strict.runtime.max_elapsed_seconds == 4 * 60 * 60
        assert loose_case.grid_refinement_tol == 1e-2
        assert strict_case.grid_refinement_tol == 1e-3


def test_dynamic_strict_refinement_reads_bar_controls_into_isolated_cpu_database():
    document = strict_refinement_document(
        name="N100_u320_low_strict_revision1_slurm_cpu",
        N=100,
        cap=320,
        settings=CAP_SETTINGS[160],
        source_database="results/bar.sqlite3",
        target_database="results/slurm.sqlite3",
        source_run_id=123,
    )

    case = document.scalar_cases()[0]
    assert case.optimizer == "peak_refinement"
    assert case.J_tol == 1e-6
    assert case.u_tol == case.v_tol == 1e-5
    assert case.projected_gradient_tol == 1e-5
    assert document.runtime.device == "cpu"
    assert document.runtime.max_elapsed_seconds == 4 * 60 * 60
    assert document.query.database == "results/bar.sqlite3"
    assert document.query.where == {
        "run_id": 123,
        "status": ["running", "complete", "failed"],
    }
    assert document.query.order_by == "best_objective"
    assert document.query.resume_optimizer is False
