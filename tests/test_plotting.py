import numpy as np
import pytest

from ofc.plotting import plot_standard_figures, save_standard_figures


def sample_run(
    run_id,
    *,
    N,
    score_shift=0.0,
    stable=True,
    r_bg=-1.5,
    u_max=10.0,
):
    score = np.asarray([1.0, 1.4, 1.7, 1.9, 2.0]) + score_shift
    objective = score + 0.4
    penalty = objective - score
    tolerance = 1e-6 if stable else 1e-2
    return {
        "run_id": run_id,
        "N": N,
        "t_interval": 1.0,
        "r_bg": r_bg,
        "slew_limit": 0.05,
        "optimizer": "adam",
        "smoothness": 1e-3,
        "u_smooth": None,
        "v_smooth": None,
        "sharpness": 0.0,
        "u_sharp": None,
        "v_sharp": None,
        "adam_learning_rate": 1e-2,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "lbfgs_history_size": 10,
        "lbfgs_max_linesearch_steps": 20,
        "lbfgs_tolerance": 1e-6,
        "u_max": u_max,
        "v_max": 20.0,
        "J_tol": 1e-5,
        "u_tol": 1e-3,
        "v_tol": 1e-3,
        "best_score": float(score[-1]),
        "best_objective": float(objective[-1]),
        "best_penalty": float(penalty[-1]),
        "history": {
            "step": np.arange(5),
            "score": score,
            "objective": objective,
            "penalty": penalty,
            "learning_rate": np.asarray([1e-2, 1e-2, 1e-2, 5e-3, 5e-3]),
            "learning_rate_change_steps": np.asarray([0, 2]),
            "stage_learning_rates": np.asarray([1e-2, 5e-3]),
        },
        "tolerances": {
            "step": np.asarray([2, 4]),
            "score_tolerance": np.asarray([1e-2, tolerance]),
            "u_tolerance": np.asarray([1e-2, tolerance]),
            "v_tolerance": np.asarray([1e-2, tolerance]),
            "passed": np.asarray([0, int(stable)]),
        },
        "controls": {
            "best": {
                "u": np.linspace(0.2 * u_max, 0.8 * u_max, N + 1),
                "v": np.linspace(-10.0, 10.0, N + 1),
            }
        },
    }


def test_standard_plot_matches_three_figure_structure():
    runs = [
        sample_run(1, N=4, stable=False),
        sample_run(2, N=4, score_shift=0.2),
        sample_run(3, N=4, score_shift=0.4),
    ]
    figures = plot_standard_figures(runs)

    assert list(figures) == ["convergence", "distribution", "controls"]
    convergence, convergence_axes = figures["convergence"]
    distribution, distribution_axis = figures["distribution"]
    controls, control_axes = figures["controls"]
    assert convergence._suptitle.get_text() == "Figure 1 — Optimisation Convergence"
    assert [axis.get_ylabel() for axis in convergence_axes] == [
        r"$J_{\mathrm{reg}}$",
        r"$J_{\mathrm{mol}}$",
        "Penalty",
    ]
    assert hasattr(convergence, "_configuration_box")
    assert distribution._suptitle.get_text() == "Figure 2 — Yield Distribution by Time-grid Size"
    assert [tick.get_text() for tick in distribution_axis.get_xticklabels()] == ["$N=4$"]
    assert controls._suptitle.get_text() == "Figure 3 — Optimised Control Overlays"
    assert [axis.get_ylabel() for axis in control_axes] == [r"$u/u_{\max}$", r"$\nu/\nu_{\max}$"]
    assert any(line.get_color() == "#c45d55" for line in control_axes[0].lines)
    assert any(line.get_color() == "#237a4b" for line in control_axes[0].lines)


def test_log_parameter_sweep_uses_best_traces_and_all_initialization_scatter():
    runs = [
        sample_run(1, N=4, u_max=1.0, score_shift=0.0),
        sample_run(2, N=4, u_max=1.0, score_shift=0.2),
        sample_run(3, N=4, u_max=10.0, score_shift=0.3),
        sample_run(4, N=4, u_max=10.0, score_shift=0.5, stable=False),
        sample_run(5, N=4, u_max=100.0, score_shift=0.6),
        sample_run(6, N=4, u_max=100.0, score_shift=0.8),
    ]

    figures = plot_standard_figures(runs)
    convergence, convergence_axes = figures["convergence"]
    distribution, distribution_axis = figures["distribution"]
    controls, control_axes = figures["controls"]

    assert convergence._sweep_parameter == "u_max"
    assert convergence._plotted_run_ids == (2, 4, 6)
    assert controls._plotted_run_ids == (2, 4, 6)
    assert distribution._scatter_run_count == len(runs)
    assert distribution_axis.get_xscale() == "log"
    assert distribution_axis.get_xlabel() == r"$u_{\max}$"
    assert "by $u_{\\max}$" in convergence._suptitle.get_text()
    assert "by $u_{\\max}$" in controls._suptitle.get_text()

    convergence_lines = convergence_axes[0].lines
    control_lines = control_axes[0].lines
    assert len(convergence_lines) == len(control_lines) == 3
    assert len({tuple(line.get_color()) for line in convergence_lines}) == 3
    for convergence_line, control_line in zip(convergence_lines, control_lines):
        np.testing.assert_allclose(
            np.asarray(convergence_line.get_color()),
            np.asarray(control_line.get_color()),
        )
    assert [line.get_alpha() for line in convergence_lines] == [1.0, 0.55, 1.0]
    assert [line.get_alpha() for line in control_lines] == [1.0, 0.55, 1.0]


def test_sweep_best_objective_and_control_are_selected_by_regularized_score():
    lower_score_higher_objective = sample_run(1, N=4, u_max=1.0)
    lower_score_higher_objective["best_objective"] = 100.0
    best_score = sample_run(2, N=4, u_max=1.0, score_shift=0.2)
    other_value = sample_run(3, N=4, u_max=10.0, score_shift=0.1)

    figures = plot_standard_figures(
        [lower_score_higher_objective, best_score, other_value]
    )
    distribution_axis = figures["distribution"][1]
    controls_figure, control_axes = figures["controls"]

    assert controls_figure._plotted_run_ids == (2, 3)
    np.testing.assert_allclose(
        distribution_axis.lines[-1].get_ydata(),
        [best_score["best_objective"], other_value["best_objective"]],
    )
    np.testing.assert_allclose(
        control_axes[0].lines[0].get_ydata(),
        best_score["controls"]["best"]["u"] / best_score["u_max"],
    )


def test_signed_log_background_sweep_uses_symmetric_log_axis():
    values = (-1.0, -0.1, -0.01, 0.01, 0.1, 1.0)
    runs = [
        sample_run(index, N=4, r_bg=value, score_shift=0.1 * index)
        for index, value in enumerate(values, start=1)
    ]

    figures = plot_standard_figures(runs)

    assert figures["distribution"][1].get_xscale() == "symlog"
    assert figures["distribution"][0]._sweep_parameter == "r_bg"


def test_multiple_parameter_sweeps_require_explicit_selection():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=8, u_max=10.0, score_shift=0.2),
    ]

    with pytest.raises(ValueError, match="vary multiple configuration parameters"):
        plot_standard_figures(runs)

    figures = plot_standard_figures(runs, sweep_parameter="u_max")
    assert figures["distribution"][0]._sweep_parameter == "u_max"


def test_standard_figures_save_as_three_separate_files(tmp_path):
    saved = save_standard_figures(
        [sample_run(1, N=4)],
        tmp_path,
        formats=("png", "pdf"),
    )

    assert {path.name for formats in saved.values() for path in formats.values()} == {
        "01_convergence.png",
        "01_convergence.pdf",
        "02_distribution.png",
        "02_distribution.pdf",
        "03_controls.png",
        "03_controls.pdf",
    }
    assert all(path.is_file() for formats in saved.values() for path in formats.values())
