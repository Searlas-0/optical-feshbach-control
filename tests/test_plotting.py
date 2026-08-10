import inspect

import numpy as np
import pytest
from matplotlib.ticker import NullFormatter, NullLocator

from ofc.plotting import (
    FIGURE_DPI,
    _draw_history_by_terminal_stability,
    _make_sweep_spec,
    _schedule_change_steps,
    _plot_sweep_convergence,
    _plot_sweep_controls,
    _plot_sweep_yield_distribution,
    plot_convergence,
    plot_controls,
    plot_double_sweep_summary,
    plot_single_sweep_summary,
    plot_standard_summary,
    plot_standard_figures,
    plot_sweep_run_summaries,
    plot_triple_sweep_summary,
    plot_yield_distribution,
    save_standard_figures,
)


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


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


def test_standard_plot_is_one_unified_summary_of_the_shown_runs():
    runs = [
        sample_run(1, N=4, stable=False),
        sample_run(2, N=4, score_shift=0.2),
        sample_run(3, N=4, score_shift=0.4),
    ]
    figures = plot_standard_figures(runs)

    assert list(figures) == ["sweep_summary"]
    figure, axes = figures["sweep_summary"]
    score_axis, objective_axis, u_axis, v_axis = axes
    assert figure._suptitle is None
    assert figure._sweep_parameter is None
    assert len(figure.axes) == 4
    assert score_axis.get_xscale() == "log"
    assert score_axis.xaxis._scale.base == 10
    assert score_axis.get_ylabel() == r"$J_{\mathrm{reg}}$"
    assert objective_axis.get_ylabel() == r"$J_{\mathrm{mol}}$"
    assert objective_axis.yaxis.get_label_position() == "right"
    np.testing.assert_allclose(score_axis.get_ylim(), objective_axis.get_ylim())
    assert score_axis._statistic_run_ids == (1, 2, 3)
    assert objective_axis._plotted_run_ids == (1, 2, 3)
    np.testing.assert_allclose(
        score_axis._median_history,
        np.median([run["history"]["score"] for run in runs], axis=0),
    )
    assert score_axis._schedule_change_steps == (2,)
    assert len(score_axis._schedule_change_lines) == 1
    assert score_axis._schedule_change_lines[0].get_linestyle() == "--"
    assert score_axis._best_score_reference_line.get_linestyle() == "--"
    assert any(r"$S_J=" in text.get_text() for text in objective_axis.texts)
    assert [u_axis.get_ylabel(), v_axis.get_ylabel()] == [r"$u$", r"$\nu$"]
    assert len(u_axis.lines) == len(v_axis.lines) == 3
    assert all(axis.get_xticks().size == 0 for axis in (u_axis, v_axis))


def test_step_markers_only_include_actual_adam_learning_rate_updates():
    adam = sample_run(1, N=4)
    adam["schedule"] = [(2, 1.0), (1, 1.0), (1, 0.5)]
    adam["history"].pop("learning_rate_change_steps")
    adam["history"].pop("stage_learning_rates")

    lbfgs = sample_run(2, N=4)
    lbfgs["optimizer"] = "lbfgs"
    lbfgs["schedule"] = [(2, 1.0), (2, 1.0)]
    lbfgs["history"].pop("learning_rate_change_steps")
    lbfgs["history"].pop("stage_learning_rates")

    assert _schedule_change_steps([adam]) == (3,)
    assert _schedule_change_steps([lbfgs]) == ()


def test_stability_blocks_do_not_split_step_history_into_checkpoint_segments():
    import matplotlib.pyplot as plt

    run = sample_run(1, N=4)
    run["tolerances"]["step"] = np.asarray([1, 2, 3, 4])
    run["tolerances"]["score_tolerance"] = np.asarray(
        [1e-2, 1e-6, 1e-2, 1e-6]
    )
    _, axis = plt.subplots()

    _draw_history_by_terminal_stability(axis, run, "score", width=5)

    assert len(axis.lines) == 1
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), np.arange(5))


def test_log_parameter_sweep_uses_best_traces_and_all_initialization_scatter():
    runs = [
        sample_run(1, N=4, u_max=1.0, score_shift=0.0),
        sample_run(2, N=4, u_max=1.0, score_shift=0.2),
        sample_run(3, N=4, u_max=10.0, score_shift=0.3),
        sample_run(4, N=4, u_max=10.0, score_shift=0.5, stable=False),
        sample_run(5, N=4, u_max=100.0, score_shift=0.6),
        sample_run(6, N=4, u_max=100.0, score_shift=0.8),
    ]

    figure, axes = plot_standard_figures(runs)["sweep_summary"]

    assert figure._sweep_parameter == "u_max"
    assert len(figure._summary_records) == 3
    assert len(axes) == 12
    assert len(figure._summary_sweep_labels) == 3
    for index, record in enumerate(figure._summary_records):
        expected_ids = (2 * index + 1, 2 * index + 2)
        assert record["run_ids"] == expected_ids
        assert record["score_statistic_run_ids"] == expected_ids
        assert record["objective"]["run_ids"] == expected_ids
        score_axis, objective_axis, u_axis, v_axis = axes[4 * index : 4 * index + 4]
        assert score_axis._statistic_run_ids == expected_ids
        assert score_axis._schedule_change_steps == (2,)
        assert len(score_axis._schedule_change_lines) == 1
        assert objective_axis._plotted_run_ids == expected_ids
        assert objective_axis.yaxis.get_label_position() == "right"
        np.testing.assert_allclose(score_axis.get_ylim(), objective_axis.get_ylim())
        assert len(u_axis.lines) == len(v_axis.lines) == 2


def test_scatter_and_line_plotters_offer_independent_optional_log_axes():
    functions = (
        plot_convergence,
        plot_yield_distribution,
        plot_controls,
        _plot_sweep_convergence,
        _plot_sweep_yield_distribution,
        _plot_sweep_controls,
    )

    for function in functions:
        parameters = inspect.signature(function).parameters
        assert parameters["log_base_x"].default is None
        assert parameters["log_base_y"].default is None
        if function in {plot_convergence, plot_yield_distribution, plot_controls}:
            assert parameters["sweep"].default is None
        expected_base_x = (
            None
            if function in {plot_yield_distribution, _plot_sweep_yield_distribution}
            else "axis"
        )
        assert parameters["base_x"].default == expected_base_x
        assert parameters["base_y"].default == "axis"
        assert parameters["x_multiplier"].default == 1.0
        assert parameters["y_multiplier"].default == 1.0
        assert parameters["x_range"].default is None
        assert parameters["y_range"].default is None
        assert parameters["x_label"].default is None
        assert parameters["y_label"].default is None
        if function in {plot_yield_distribution, _plot_sweep_yield_distribution}:
            assert parameters["seed_sensitivity_log_base_y"].default == 10
            assert parameters["seed_sensitivity_base_y"].default is None
            assert parameters["seed_sensitivity_y_multiplier"].default == 1.0
            assert parameters["seed_sensitivity_y_range"].default is None


def test_sweep_distribution_uses_categorical_x_and_configurable_log_y_style():
    runs = [
        sample_run(1, N=4, u_max=1.0, score_shift=0.0),
        sample_run(2, N=4, u_max=1.0, score_shift=0.2),
        sample_run(3, N=4, u_max=10.0, score_shift=0.3),
        sample_run(4, N=4, u_max=10.0, score_shift=0.5),
        sample_run(5, N=4, u_max=100.0, score_shift=0.6),
        sample_run(6, N=4, u_max=100.0, score_shift=0.8),
    ]
    sweep = _make_sweep_spec(runs, "u_max")

    figure, axis = _plot_sweep_yield_distribution(
        runs,
        sweep,
        log_base_x=10,
        log_base_y=10,
        base_x=None,
        base_y=None,
        point_size=12,
        line_alpha=0.22,
    )

    assert axis.get_xscale() == "linear"
    assert axis.get_yscale() == "log"
    assert axis.yaxis._scale.base == 10
    assert axis.lines[0].get_alpha() == pytest.approx(0.22)
    assert all(
        collection.get_sizes()[0] == pytest.approx(12)
        for collection in axis.collections[:-1]
    )
    assert axis.collections[-1].get_sizes()[0] == pytest.approx(16.8)
    assert isinstance(axis.yaxis.get_minor_formatter(), NullFormatter)
    np.testing.assert_allclose(axis.get_xlim(), [-0.38, 2.38])
    assert axis.get_ylim()[0] > 0.0
    np.testing.assert_allclose(axis.get_xticks(), [0.0, 1.0, 2.0])
    assert [
        tick.get_text()
        for tick in figure._seed_sensitivity_axis.get_xticklabels()
    ] == ["1", "10", "100"]
    assert len(figure.axes) == 2
    first_decade = np.arange(1.0, 10.01, 0.25)
    second_decade = 10.0 * first_decade
    first_pixels = np.asarray(
        [axis.transData.transform((1.0, value))[1] for value in first_decade]
    )
    second_pixels = np.asarray(
        [axis.transData.transform((1.0, value))[1] for value in second_decade]
    )
    first_gaps = np.diff(first_pixels)
    second_gaps = np.diff(second_pixels)
    assert np.all(np.diff(first_gaps) < 0.0)
    np.testing.assert_allclose(
        first_gaps,
        second_gaps,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        axis.yaxis.get_minorticklocs(),
        np.arange(2.5, 10.0, 0.25),
    )
    figure.canvas.draw()
    minor_lengths = {
        float(location): tick.tick1line.get_markersize()
        for location, tick in zip(
            axis.yaxis.get_minorticklocs(), axis.yaxis.get_minor_ticks()
        )
    }
    assert minor_lengths[2.5] > minor_lengths[2.75]
    assert axis.xaxis.get_minorticklocs().size == 0
    assert axis.figure.dpi == FIGURE_DPI


def test_distribution_keeps_plain_categorical_x_labels_and_configurable_log_y():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=4, u_max=10.0, score_shift=0.2),
        sample_run(3, N=4, u_max=100.0, score_shift=0.4),
    ]
    sweep = _make_sweep_spec(runs, "u_max")

    _, actual_value_axis = _plot_sweep_yield_distribution(
        runs,
        sweep,
        log_base_x=10,
        log_base_y=10,
    )

    assert [
        tick.get_text()
        for tick in actual_value_axis.figure._seed_sensitivity_axis.get_xticklabels()
    ] == [
        "1",
        "10",
        "100",
    ]

    np.testing.assert_allclose(actual_value_axis.get_xticks(), [0.0, 1.0, 2.0])
    assert actual_value_axis.get_xscale() == "linear"

    _, multiplied_axis = _plot_sweep_yield_distribution(
        runs,
        sweep,
        log_base_y=10,
        base_y="axis",
        y_multiplier=2,
    )
    assert 2.0 in multiplied_axis.get_yticks()
    formatter = multiplied_axis.yaxis.get_major_formatter()
    assert formatter(2.0, 0) == "2"
    assert formatter(20.0, 0) == "20"
    assert formatter(-20.0, 0) == "-20"

    _, fractional_axis = _plot_sweep_yield_distribution(
        runs,
        sweep,
        log_base_y=10,
        base_y=None,
        y_multiplier=0.1,
    )
    assert 0.1 in fractional_axis.get_yticks()
    assert fractional_axis.yaxis.get_major_formatter()(0.1, 0) == "0.1"


def test_strip_sweep_values_are_equally_spaced_and_have_plain_numeric_labels():
    values = (6.25e-8, 1.25e-7, 1e-6, 2e-6, 1e-4, 0.25)
    runs = []
    for run_id, value in enumerate(values, start=1):
        run = sample_run(run_id, N=4, score_shift=0.1 * run_id)
        run["smoothness"] = value
        runs.append(run)

    sweep = _make_sweep_spec(runs, "smoothness")
    figure, _ = _plot_sweep_yield_distribution(runs, sweep)
    axis = figure._seed_sensitivity_axis
    figure.canvas.draw()

    np.testing.assert_allclose(axis.get_xticks(), np.arange(len(values)))
    labels = [label.get_text() for label in axis.get_xticklabels()]
    assert labels == ["6.25e-8", "1.25e-7", "1e-6", "2e-6", "1e-4", "0.25"]
    assert all("$" not in label and "\\" not in label for label in labels)
    pixels = axis.transData.transform(
        np.column_stack((axis.get_xticks(), np.zeros(len(values))))
    )[:, 0]
    np.testing.assert_allclose(np.diff(pixels), np.diff(pixels)[0], rtol=1e-6)
    np.testing.assert_allclose(axis.get_xlim(), [-0.38, 5.38])


def test_log_range_collapses_its_floor_to_zero_and_tightens_to_data():
    objective_values = (1e-4, 1e-3, 1.0, 100.0)
    runs = [
        sample_run(index, N=4, u_max=10.0**index)
        for index in range(len(objective_values))
    ]
    for run, objective in zip(runs, objective_values):
        run["best_objective"] = objective
    sweep = _make_sweep_spec(runs, "u_max")

    figure, axis = _plot_sweep_yield_distribution(
        runs,
        sweep,
        log_base_y=10,
        base_y=None,
        y_range=(1e-3, 1000.0),
    )
    figure.canvas.draw()

    assert axis.yaxis._scale.cutoff == pytest.approx(1e-3)
    np.testing.assert_allclose(axis.get_ylim(), [1e-3, 100.0])
    np.testing.assert_allclose(
        axis.get_yticks(),
        [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    )
    assert axis.yaxis.get_major_formatter()(1e-3, 0) == "0"
    zero_pixel = axis.transData.transform((1.0, 0.0))[1]
    assert axis.transData.transform((1.0, 1e-4))[1] == pytest.approx(zero_pixel)
    assert axis.transData.transform((1.0, 1e-3))[1] == pytest.approx(zero_pixel)


def test_log_range_uses_the_smallest_plotted_value_as_zero_when_it_is_higher():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=4, u_max=10.0),
        sample_run(3, N=4, u_max=100.0),
    ]
    for run, objective in zip(runs, (1.0, 10.0, 100.0)):
        run["best_objective"] = objective
    sweep = _make_sweep_spec(runs, "u_max")

    _, axis = _plot_sweep_yield_distribution(
        runs,
        sweep,
        log_base_y=10,
        base_y=None,
        y_range=(1e-3, 1000.0),
    )

    assert axis.yaxis._scale.cutoff == pytest.approx(1.0)
    np.testing.assert_allclose(axis.get_ylim(), [1.0, 100.0])
    np.testing.assert_allclose(axis.get_yticks(), [1.0, 10.0, 100.0])
    assert [axis.yaxis.get_major_formatter()(value, 0) for value in axis.get_yticks()] == [
        "0",
        "10",
        "100",
    ]


def test_log_axis_ends_at_next_major_tick_without_exceeding_its_range_cap():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=4, u_max=10.0),
    ]
    for run, objective in zip(runs, (1e-3, 70.0)):
        run["best_objective"] = objective
    sweep = _make_sweep_spec(runs, "u_max")

    _, rounded_axis = _plot_sweep_yield_distribution(
        runs,
        sweep,
        log_base_y=10,
        base_y=None,
        y_range=(1e-3, 1000.0),
    )
    np.testing.assert_allclose(rounded_axis.get_ylim(), [1e-3, 100.0])
    assert 100.0 in rounded_axis.get_yticks()

    _, capped_axis = _plot_sweep_yield_distribution(
        runs,
        sweep,
        log_base_y=10,
        base_y=None,
        y_range=(1e-3, 80.0),
    )
    np.testing.assert_allclose(capped_axis.get_ylim(), [1e-3, 80.0])
    assert 80.0 in capped_axis.get_yticks()
    assert capped_axis.yaxis.get_major_formatter()(80.0, 0) == "80"


def test_distribution_reports_neutral_percentile_spread_and_seed_sensitivity():
    runs = [
        sample_run(1, N=4, u_max=1.0, score_shift=0.0),
        sample_run(2, N=4, u_max=1.0, score_shift=0.2),
        sample_run(3, N=4, u_max=10.0, score_shift=0.0),
        sample_run(4, N=4, u_max=10.0, score_shift=1.0),
    ]
    for run in runs:
        run["J_tol"] = 0.1
    sweep = _make_sweep_spec(runs, "u_max")

    figure, distribution_axis = _plot_sweep_yield_distribution(runs, sweep)
    sensitivity_axis = figure._seed_sensitivity_axis
    records = figure._seed_sensitivity

    assert len(figure.axes) == 2
    assert sensitivity_axis.get_ylabel() == r"Seed sensitivity $S_J$"
    assert figure._seed_sensitivity_tolerances == (0.1,)
    assert [record[0] for record in records] == [1.0, 10.0]
    expected = []
    for values in ([2.4, 2.6], [2.4, 3.4]):
        lower, median, upper = np.percentile(values, [10.0, 50.0, 90.0])
        expected.append((upper - lower) / abs(median))
    np.testing.assert_allclose([record[1] for record in records], expected)
    np.testing.assert_allclose(
        [record[2] for record in records],
        [np.percentile([2.4, 2.6], 10), np.percentile([2.4, 3.4], 10)],
    )
    np.testing.assert_allclose(
        [record[3] for record in records],
        [np.percentile([2.4, 2.6], 50), np.percentile([2.4, 3.4], 50)],
    )
    np.testing.assert_allclose(
        [record[4] for record in records],
        [np.percentile([2.4, 2.6], 90), np.percentile([2.4, 3.4], 90)],
    )
    assert sensitivity_axis.get_yscale() == "log"
    assert sensitivity_axis.yaxis._scale.base == 10
    cutoff = sensitivity_axis.yaxis._scale.cutoff
    assert cutoff == pytest.approx(min(expected))
    figure.canvas.draw()
    zero_pixel = sensitivity_axis.transData.transform((1.0, 0.0))[1]
    assert sensitivity_axis.transData.transform((1.0, cutoff))[1] == pytest.approx(
        zero_pixel
    )
    threshold_lines = [
        line
        for line in sensitivity_axis.lines
        if np.asarray(line.get_ydata()).size == 2
        and np.allclose(line.get_ydata(), [0.1, 0.1])
    ]
    assert len(threshold_lines) == 1
    assert not threshold_lines[0].get_clip_on()
    assert threshold_lines[0].get_zorder() > sensitivity_axis.spines["bottom"].get_zorder()
    assert len(distribution_axis.lines) == 3 * len(records)
    connector, endpoints, median = distribution_axis.lines[:3]
    assert connector.get_linewidth() == pytest.approx(0.9)
    assert endpoints.get_linewidth() == pytest.approx(1.25)
    assert median.get_linewidth() == pytest.approx(1.0)
    assert all(line.get_alpha() == pytest.approx(0.22) for line in distribution_axis.lines)
    assert all(line.get_color() == "#202020" for line in distribution_axis.lines)
    group_x = distribution_axis.collections[0].get_offsets()[:, 0]
    endpoint_x = np.asarray(endpoints.get_xdata(), dtype=float)
    endpoint_x = endpoint_x[np.isfinite(endpoint_x)]
    median_x = np.asarray(median.get_xdata(), dtype=float)
    assert endpoint_x.min() < group_x.min()
    assert endpoint_x.max() > group_x.max()
    assert endpoint_x.min() < median_x.min()
    assert endpoint_x.max() > median_x.max()
    group_width = group_x.max() - group_x.min()
    endpoint_width = endpoint_x.max() - endpoint_x.min()
    assert endpoint_width > 1.5 * group_width
    assert [text.get_text() for text in distribution_axis.get_legend().get_texts()] == [
        "All initializations",
        "Best score at each value",
    ]


def test_seed_tolerance_below_log_cutoff_remains_visible_on_the_zero_boundary():
    runs = [
        sample_run(1, N=4, u_max=1.0, score_shift=0.0),
        sample_run(2, N=4, u_max=1.0, score_shift=0.2),
        sample_run(3, N=4, u_max=10.0, score_shift=0.4),
        sample_run(4, N=4, u_max=10.0, score_shift=0.8),
    ]
    sweep = _make_sweep_spec(runs, "u_max")

    figure, _ = _plot_sweep_yield_distribution(
        runs,
        sweep,
        seed_sensitivity_y_range=(1e-3, 100.0),
    )
    axis = figure._seed_sensitivity_axis
    figure.canvas.draw()
    tolerance = runs[0]["J_tol"]
    tolerance_line = next(
        line
        for line in axis.lines
        if np.asarray(line.get_ydata()).size == 2
        and np.allclose(line.get_ydata(), [tolerance, tolerance])
    )

    assert axis.yaxis._scale.cutoff == pytest.approx(1e-3)
    assert tolerance_line.get_visible()
    assert not tolerance_line.get_clip_on()
    assert tolerance_line.get_zorder() > axis.spines["bottom"].get_zorder()
    assert axis.transData.transform((1.0, tolerance))[1] == pytest.approx(
        axis.bbox.y0
    )


def test_seed_sensitivity_scale_is_independent_and_uses_all_database_tolerances():
    runs = [
        sample_run(1, N=4, u_max=1.0, score_shift=0.0),
        sample_run(2, N=4, u_max=1.0, score_shift=0.2),
        sample_run(3, N=4, u_max=10.0, score_shift=0.4),
        sample_run(4, N=4, u_max=10.0, score_shift=0.8),
    ]
    for run, tolerance in zip(runs, (0.05, 0.1, 0.05, 0.1)):
        run["J_tol"] = tolerance
    sweep = _make_sweep_spec(runs, "u_max")

    figure, objective_axis = _plot_sweep_yield_distribution(
        runs,
        sweep,
        log_base_y=10,
        seed_sensitivity_log_base_y=2,
        seed_sensitivity_base_y=None,
        seed_sensitivity_y_multiplier=0.5,
        seed_sensitivity_y_range=(0.01, 1.0),
    )
    sensitivity_axis = figure._seed_sensitivity_axis

    assert objective_axis.yaxis._scale.base == 10
    assert sensitivity_axis.yaxis._scale.base == 2
    assert figure._seed_sensitivity_tolerances == (0.05, 0.1)
    for tolerance in figure._seed_sensitivity_tolerances:
        assert any(
            np.allclose(line.get_ydata(), [tolerance, tolerance])
            for line in sensitivity_axis.lines
        )


def test_seed_sensitivity_tolerance_can_override_or_hide_stored_values():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=4, u_max=1.0, score_shift=0.2),
        sample_run(3, N=4, u_max=10.0, score_shift=0.4),
        sample_run(4, N=4, u_max=10.0, score_shift=0.8),
    ]
    sweep = _make_sweep_spec(runs, "u_max")

    overridden, _ = _plot_sweep_yield_distribution(
        runs,
        sweep,
        seed_sensitivity_tolerance=[0.2, 0.05],
    )
    hidden, _ = _plot_sweep_yield_distribution(
        runs,
        sweep,
        seed_sensitivity_tolerance=[],
    )

    assert overridden._seed_sensitivity_tolerances == (0.05, 0.2)
    assert hidden._seed_sensitivity_tolerances == ()


def test_sweep_convergence_positive_log_x_starts_at_base_power_zero():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=4, u_max=10.0, score_shift=0.2),
    ]
    sweep = _make_sweep_spec(runs, "u_max")

    _, axes = _plot_sweep_convergence(runs, sweep, log_base_x=10)

    assert all(axis.get_xscale() == "log" for axis in axes)
    assert axes[-1].get_xlim()[0] == pytest.approx(1.0)
    assert axes[-1].get_xlim()[1] == pytest.approx(10.0)
    assert axes[-1].xaxis._scale.base == 10
    assert 1.0 in axes[-1].get_xticks()
    colourbar_axis = axes[-1].figure.axes[-1]
    assert colourbar_axis.get_position().y0 == pytest.approx(
        axes[-1].get_position().y0, abs=1e-3
    )
    assert colourbar_axis.get_position().y1 == pytest.approx(
        axes[0].get_position().y1, abs=1e-3
    )
    assert isinstance(colourbar_axis.yaxis.get_minor_locator(), NullLocator)


def test_convergence_log_range_uses_step_one_as_zero_and_ends_at_next_major_tick():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=4, u_max=10.0, score_shift=0.2),
    ]
    sweep = _make_sweep_spec(runs, "u_max")

    figure, axes = _plot_sweep_convergence(
        runs,
        sweep,
        log_base_x=10,
        base_x=None,
        x_range=(1e-3, 1000.0),
    )
    figure.canvas.draw()

    for axis in axes:
        assert axis.xaxis._scale.cutoff == pytest.approx(1.0)
        np.testing.assert_allclose(axis.get_xlim(), [1.0, 10.0])
        assert 10.0 in axis.get_xticks()
        assert axis.xaxis.get_major_formatter()(1.0, 0) == "0"
        assert axis.transData.transform((0.0, 1.0))[0] == pytest.approx(
            axis.transData.transform((1.0, 1.0))[0]
        )


def test_sweep_convergence_uses_equal_spacing_and_narrow_panel_gutters():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=4, u_max=10.0, score_shift=0.2),
    ]
    sweep = _make_sweep_spec(runs, "u_max")

    _, axes = _plot_sweep_convergence(runs, sweep)

    pixel_intervals = []
    for axis in axes:
        ticks = axis.get_yticks()
        first = axis.transData.transform((0.0, ticks[0]))[1]
        second = axis.transData.transform((0.0, ticks[1]))[1]
        pixel_intervals.append(second - first)
    assert pixel_intervals == pytest.approx(
        [pixel_intervals[0]] * 3, rel=1e-10, abs=1e-10
    )

    for upper_axis, lower_axis in zip(axes[:-1], axes[1:]):
        upper_bottom = upper_axis.transData.transform(
            (0.0, upper_axis.get_yticks()[0])
        )[1]
        lower_top_label = lower_axis.transData.transform(
            (0.0, lower_axis.get_yticks()[-1])
        )[1]
        gap = upper_bottom - lower_top_label
        assert pixel_intervals[0] * 0.1 < gap < pixel_intervals[0] * 0.35
    assert not axes[0].spines["bottom"].get_visible()
    assert not axes[1].spines["bottom"].get_visible()
    assert axes[2].spines["bottom"].get_visible()
    assert not axes[0].yaxis.get_major_ticks()[0].label1.get_visible()
    assert not axes[1].yaxis.get_major_ticks()[0].label1.get_visible()
    assert axes[2].yaxis.get_major_ticks()[0].label1.get_visible()
    for lower_axis in axes[1:]:
        assert lower_axis.spines["top"].get_visible()
        assert lower_axis.spines["top"].get_position() == (
            "data",
            lower_axis.get_yticks()[-1],
        )


def test_sweep_convergence_separators_remain_at_panel_joins_on_log_y_scale():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=4, u_max=10.0, score_shift=0.2),
    ]
    sweep = _make_sweep_spec(runs, "u_max")

    figure, axes = _plot_sweep_convergence(
        runs,
        sweep,
        log_base_y=10,
        y_multiplier=0.1,
        y_range=(0.1, 10.0),
    )
    figure.canvas.draw()

    for upper_axis, lower_axis in zip(axes[:-1], axes[1:]):
        assert lower_axis.spines["top"].get_visible()
        assert lower_axis.spines["top"].get_position() == ("axes", 1.0)
        separator_y = lower_axis.spines["top"].get_window_extent().y0
        gap = upper_axis.bbox.y0 - separator_y
        assert 0.0 < gap < figure.bbox.height * 0.01


def test_convergence_accepts_one_y_multiplier_per_panel_for_subunit_ticks():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=4, u_max=10.0, score_shift=0.2),
    ]
    sweep = _make_sweep_spec(runs, "u_max")

    _, axes = _plot_sweep_convergence(
        runs,
        sweep,
        log_base_y=2,
        y_multiplier=(1.0, 0.01, 0.001),
    )

    assert 1.0 in axes[0].get_yticks()
    assert 0.01 in axes[1].get_yticks()
    assert 0.001 in axes[2].get_yticks()

    with pytest.raises(ValueError, match="one value per panel"):
        _plot_sweep_convergence(
            runs,
            sweep,
            log_base_y=2,
            y_multiplier=(1.0, 0.01),
        )


def test_sweep_convergence_panel_layout_includes_negative_start_and_zero():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=4, u_max=10.0, score_shift=0.2),
    ]
    for offset, run in enumerate(runs):
        run["history"]["score"] = np.asarray([-3.2, -1.1, -0.2, 0.1, 0.4]) + offset
        run["history"]["objective"] = np.asarray([-2.4, -0.8, -0.1, 0.3, 0.7]) + offset
        run["history"]["penalty"] = np.asarray([80.0, 140.0, 210.0, 260.0, 290.0])
    sweep = _make_sweep_spec(runs, "u_max")

    _, axes = _plot_sweep_convergence(runs, sweep)

    for axis in axes[:2]:
        assert axis.get_yticks()[0] < 0.0
        assert 0.0 in axis.get_yticks()
        assert axis.get_ylim()[0] == pytest.approx(axis.get_yticks()[0])
    assert axes[2].get_yticks()[-1] == pytest.approx(300.0)
    assert axes[2].spines["top"].get_position() == ("data", 300.0)


def test_log_control_axes_use_positive_log_or_signed_log_from_the_data():
    _, axes = plot_controls(
        [sample_run(1, N=4)],
        log_base_y=10,
    )

    assert axes[0].get_yscale() == "log"
    assert axes[1].get_yscale() == "symlog"
    assert axes[0].get_ylim()[0] > 0.0
    assert axes[1].get_ylim()[0] < 0.0
    assert axes[1].get_ylim()[1] > 0.0
    assert axes[1].yaxis.get_major_formatter()(-10.0, 0) == "-10"
    assert axes[1].yaxis.get_major_formatter()(0.0, 0) == "0"
    assert axes[1].yaxis.get_major_formatter()(10.0, 0) == "10"


def test_base_two_axes_keep_subunit_minor_ticks_and_signed_central_ticks():
    _, axes = plot_controls(
        [sample_run(1, N=4)],
        log_base_y=2,
    )

    positive_minor_ticks = axes[0].get_yticks(minor=True)
    assert 0.25 in positive_minor_ticks
    assert 0.5 in positive_minor_ticks
    central_minor_ticks = axes[1].get_yticks(minor=True)
    np.testing.assert_allclose(
        central_minor_ticks[(central_minor_ticks > -1.0) & (central_minor_ticks < 1.0)],
        [-0.5, 0.5],
    )


def test_signed_log_range_applies_the_same_absolute_cutoff_in_both_directions():
    run = sample_run(1, N=4)
    run["controls"]["best"]["v"] = np.asarray(
        [-2000.0, -0.02, 0.0, 0.02, 2000.0]
    )

    figure, axes = plot_controls(
        [run],
        log_base_y=10,
        y_range=(1e-3, 1000.0),
    )
    axis = axes[1]
    figure.canvas.draw()

    assert axis.get_yscale() == "symlog"
    assert axis.yaxis._scale.cutoff == pytest.approx(1e-3)
    np.testing.assert_allclose(axis.get_ylim(), [-100.0, 100.0])
    zero_pixel = axis.transData.transform((0.0, 0.0))[1]
    for value in (-1e-3, -1e-4, 0.0, 1e-4, 1e-3):
        assert axis.transData.transform((0.0, value))[1] == pytest.approx(
            zero_pixel
        )


def test_controls_only_label_zero_and_the_final_time_for_regular_and_sweep_views():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=4, u_max=10.0, score_shift=0.2),
    ]
    runs[0]["t_interval"] = 0.75
    runs[1]["t_interval"] = 4.0

    _, regular_axes = plot_controls(runs)
    sweep = _make_sweep_spec(runs, "u_max")
    _, sweep_axes = _plot_sweep_controls(runs, sweep)

    for axis in (*regular_axes, *sweep_axes):
        ticks = axis.get_xticks()
        assert 0.0 in ticks
        assert 4.0 in ticks
        assert np.any((ticks > 0.0) & (ticks < 4.0))
        labels = [tick.get_text() for tick in axis.get_xticklabels()]
        assert [label for label in labels if label] == ["0.0", "4.0"]
        assert any(label == "" for label in labels)


def test_distribution_keeps_categorical_x_with_configurable_y_range_and_labels():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=4, u_max=10.0, score_shift=0.2),
    ]
    sweep = _make_sweep_spec(runs, "u_max")

    _, axis = _plot_sweep_yield_distribution(
        runs,
        sweep,
        x_range=(0.8, 12.0),
        y_range=(2.0, 4.0),
        x_label=r"$U_{\max}$",
        y_label=r"$\mathcal{J}$",
    )
    np.testing.assert_allclose(axis.get_xlim(), [-0.38, 1.38])
    np.testing.assert_allclose(axis.get_ylim(), [2.0, 4.0])
    assert axis.figure._seed_sensitivity_axis.get_xlabel() == r"$U_{\max}$"
    assert axis.get_ylabel() == r"$\mathcal{J}$"

    _, axes = plot_controls(
        runs,
        x_range=(0.0, 0.5),
        y_range=(-0.75, 0.75),
        x_label=r"$t/T$",
        y_label=(r"$\tilde u$", r"$\tilde\nu$"),
    )
    for axis, ylabel in zip(axes, (r"$\tilde u$", r"$\tilde\nu$")):
        np.testing.assert_allclose(axis.get_xlim(), [0.0, 0.5])
        np.testing.assert_allclose(axis.get_ylim(), [-0.75, 0.75])
        assert axis.get_xlabel() == r"$t/T$"
        assert axis.get_ylabel() == ylabel


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"log_base_y": 1}, "log_base_y"),
        ({"base_y": 1}, "base_y"),
        ({"x_multiplier": 0}, "x_multiplier"),
        ({"y_multiplier": -1}, "y_multiplier"),
        ({"point_size": 0}, "point_size"),
        ({"line_alpha": 1.1}, "line_alpha"),
        ({"seed_sensitivity_log_base_y": 1}, "seed_sensitivity_log_base_y"),
        ({"seed_sensitivity_y_range": (1.0,)}, "seed_sensitivity_y_range"),
        ({"x_range": (2.0, 1.0)}, "x_range"),
        ({"y_range": (1.0,)}, "y_range"),
        ({"x_label": ("one", "two")}, "x_label"),
    ],
)
def test_sweep_distribution_rejects_invalid_style_values(kwargs, message):
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=4, u_max=10.0),
    ]
    sweep = _make_sweep_spec(runs, "u_max")

    with pytest.raises(ValueError, match=message):
        _plot_sweep_yield_distribution(runs, sweep, **kwargs)


def test_sweep_best_objective_and_control_are_selected_by_regularized_score():
    lower_score_higher_objective = sample_run(1, N=4, u_max=1.0)
    lower_score_higher_objective["best_objective"] = 100.0
    best_score = sample_run(2, N=4, u_max=1.0, score_shift=0.2)
    other_value = sample_run(3, N=4, u_max=10.0, score_shift=0.1)

    figures = plot_standard_figures(
        [lower_score_higher_objective, best_score, other_value]
    )
    figure, axes = figures["sweep_summary"]
    first_record = figure._summary_records[0]

    assert first_record["objective"]["run_ids"] == (1, 2)
    assert first_record["objective"]["best_run_id"] == 2
    np.testing.assert_allclose(
        first_record["objective"]["values"],
        [lower_score_higher_objective["best_objective"], best_score["best_objective"]],
    )
    np.testing.assert_allclose(
        axes[2].lines[1].get_ydata(),
        best_score["controls"]["best"]["u"],
    )


def test_compact_sweep_summary_keeps_only_score_and_raw_controls_with_stability():
    runs = [
        sample_run(1, N=4, u_max=10.0, score_shift=0.0, stable=True),
        sample_run(2, N=4, u_max=10.0, score_shift=0.2, stable=False),
        sample_run(3, N=4, u_max=100.0, score_shift=0.4, stable=True),
        sample_run(4, N=4, u_max=100.0, score_shift=0.6, stable=True),
    ]
    by_id = {run["run_id"]: run for run in runs}
    sweep = _make_sweep_spec(runs, "u_max")

    figure = plot_sweep_run_summaries(
        runs,
        sweep,
        load_history=lambda run_id: by_id[run_id]["history"],
        load_tolerances=lambda run_id: by_id[run_id]["tolerances"],
        load_controls=lambda run_id: by_id[run_id]["controls"]["best"],
        history_points=4,
    )

    assert len(figure.axes) == 8
    score_axes = [axis for axis in figure.axes if axis.get_ylabel() == r"$J_{\mathrm{reg}}$"]
    assert len(score_axes) == 2
    objective_axes = [
        axis for axis in figure.axes if axis.get_ylabel() == r"$J_{\mathrm{mol}}$"
    ]
    assert len(objective_axes) == 2
    assert all(axis.get_xscale() == "log" for axis in score_axes)
    assert all(axis._schedule_change_steps == (2,) for axis in score_axes)
    assert all(len(axis._schedule_change_lines) == 1 for axis in score_axes)
    assert all(axis.xaxis._scale.cutoff == pytest.approx(1.0) for axis in score_axes)
    assert all(axis.xaxis.get_label_position() == "top" for axis in score_axes)
    assert all(not hasattr(axis, "_score_reference_axis") for axis in score_axes)
    assert all(
        objective._plotted_run_ids == score._statistic_run_ids
        for score, objective in zip(score_axes, objective_axes)
    )
    assert all(
        np.allclose(score.get_ylim(), objective.get_ylim())
        for score, objective in zip(score_axes, objective_axes)
    )
    first_u_axis = next(axis for axis in figure.axes if axis.get_ylabel() == r"$u$")
    first_v_axis = next(axis for axis in figure.axes if axis.get_ylabel() == r"$\nu$")
    np.testing.assert_allclose(
        first_u_axis.lines[0].get_ydata(),
        runs[0]["controls"]["best"]["u"],
    )
    np.testing.assert_allclose(first_u_axis.get_ylim(), [0.0, 10.0])
    assert first_u_axis.yaxis.get_label_position() == "left"
    assert first_v_axis.yaxis.get_label_position() == "right"
    sampled_steps = np.asarray([0, 1, 2, 4])
    score_lines = [
        np.asarray(line.get_ydata(), dtype=float)
        for line in score_axes[0].lines
        if len(line.get_ydata()) == len(sampled_steps)
    ]
    for run in runs[:2]:
        assert any(
            np.allclose(values, run["history"]["score"][sampled_steps])
            for values in score_lines
        )
    for control_axis in (first_u_axis, first_v_axis):
        assert control_axis.get_xlabel() == ""
        assert control_axis.get_xticks().size == 0
        np.testing.assert_allclose(control_axis.lines[0].get_xdata(), np.linspace(0, 1, 5))
    assert len(figure._summary_sweep_labels) == 2
    assert all(
        text.get_rotation() == pytest.approx(90.0)
        for text in figure._summary_sweep_labels
    )
    assert not any("Best score:" in text.get_text() for axis in score_axes for text in axis.texts)
    assert figure._summary_records[0]["best_score"] == pytest.approx(
        runs[1]["best_score"]
    )
    assert figure._summary_records[0]["median_best_score"] == pytest.approx(
        np.median([runs[0]["best_score"], runs[1]["best_score"]])
    )
    assert figure._summary_records[0]["score_stable"] == 1
    assert figure._summary_records[0]["controls"]["u"]["stable"] == 1


def test_summary_statistics_cannot_include_runs_outside_the_displayed_selection():
    shown = [
        sample_run(1, N=4, u_max=10.0, score_shift=0.0),
        sample_run(2, N=4, u_max=10.0, score_shift=0.2),
    ]
    outside = sample_run(99, N=4, u_max=10.0, score_shift=100.0)
    by_id = {run["run_id"]: run for run in (*shown, outside)}

    figure = plot_single_sweep_summary(
        shown,
        _make_sweep_spec(shown, "u_max", allow_single=True),
        load_history=lambda run_id: by_id[run_id]["history"],
        load_tolerances=lambda run_id: by_id[run_id]["tolerances"],
        load_controls=lambda run_id: by_id[run_id]["controls"]["best"],
    )
    score_axis, objective_axis, _, _ = figure._summary_axes

    assert score_axis._statistic_run_ids == (1, 2)
    assert objective_axis._plotted_run_ids == (1, 2)
    np.testing.assert_allclose(
        score_axis._median_history,
        np.median([run["history"]["score"] for run in shown], axis=0),
    )
    assert figure._summary_records[0]["objective"]["run_ids"] == (1, 2)


def test_double_sweep_summary_colours_best_combination_runs_and_repeats_rectangles():
    runs = []
    run_id = 1
    for u_max in (10.0, 100.0):
        for learning_rate in (1e-3, 1e-2):
            for seed in range(2):
                run = sample_run(
                    run_id,
                    N=4,
                    u_max=u_max,
                    score_shift=0.2 * run_id,
                )
                run["adam_learning_rate"] = learning_rate
                runs.append(run)
                run_id += 1
    by_id = {run["run_id"]: run for run in runs}

    figure = plot_double_sweep_summary(
        runs,
        separate_sweep=_make_sweep_spec(runs, "u_max"),
        colour_sweep=_make_sweep_spec(runs, "adam_learning_rate"),
        load_history=lambda run_id: by_id[run_id]["history"],
        load_tolerances=lambda run_id: by_id[run_id]["tolerances"],
        load_controls=lambda run_id: by_id[run_id]["controls"]["best"],
        history_points=4,
    )

    assert figure._separate_sweep_parameter == "u_max"
    assert figure._colour_sweep_parameter == "adam_learning_rate"
    assert len(figure._summary_records) == 2
    assert len(figure._summary_axes) == 8
    assert all(len(record["run_ids"]) == 2 for record in figure._summary_records)
    for index, record in enumerate(figure._summary_records):
        score_axis = figure._summary_axes[4 * index]
        assert score_axis._statistic_run_ids == record["run_ids"]
        expected = np.median(
            [by_id[run_id]["history"]["score"][[0, 1, 2, 4]] for run_id in record["run_ids"]],
            axis=0,
        )
        np.testing.assert_allclose(score_axis._median_history, expected)
    assert figure._summary_colourbar.ax in figure.axes
    assert len(figure._summary_colourbars) == 2
    figure.canvas.draw()
    for index, colourbar in enumerate(figure._summary_colourbars):
        u_axis = figure._summary_axes[4 * index + 2]
        assert colourbar.ax.get_position().y1 == pytest.approx(
            u_axis.get_position().y0
        )
        assert all(
            "$" not in label.get_text() and "\\" not in label.get_text()
            for label in colourbar.ax.get_xticklabels()
        )


def test_triple_sweep_summary_uses_matrix_cells_and_only_best_score_reference():
    runs = []
    run_id = 1
    for u_max in (10.0, 100.0):
        for smoothness in (1e-4, 1e-3):
            for learning_rate in (1e-3, 1e-2):
                run = sample_run(
                    run_id,
                    N=4,
                    u_max=u_max,
                    score_shift=0.1 * run_id,
                )
                run["smoothness"] = smoothness
                run["adam_learning_rate"] = learning_rate
                runs.append(run)
                run_id += 1
    by_id = {run["run_id"]: run for run in runs}

    figure = plot_triple_sweep_summary(
        runs,
        row_sweep=_make_sweep_spec(runs, "u_max"),
        column_sweep=_make_sweep_spec(runs, "smoothness"),
        colour_sweep=_make_sweep_spec(runs, "adam_learning_rate"),
        load_history=lambda run_id: by_id[run_id]["history"],
        load_tolerances=lambda run_id: by_id[run_id]["tolerances"],
        load_controls=lambda run_id: by_id[run_id]["controls"]["best"],
        history_points=4,
    )

    assert figure._row_sweep_parameter == "u_max"
    assert figure._column_sweep_parameter == "smoothness"
    assert figure._colour_sweep_parameter == "adam_learning_rate"
    assert len(figure._summary_records) == 4
    score_axes = list(figure._summary_axes[0::4])
    objective_axes = list(figure._summary_axes[1::4])
    assert all(len(axis._score_reference_values) == 1 for axis in score_axes)
    assert all("score_stable" not in record for record in figure._summary_records)
    assert [axis.get_ylabel() for axis in score_axes] == [
        r"$J_{\mathrm{reg}}$",
        "",
        r"$J_{\mathrm{reg}}$",
        "",
    ]
    assert [axis.get_xlabel() for axis in score_axes] == [
        "Optimisation step",
        "Optimisation step",
        "",
        "",
    ]
    figure.canvas.draw()
    assert all(any(label.get_visible() for label in axis.get_xticklabels()) for axis in score_axes)
    assert all(any(label.get_visible() for label in axis.get_yticklabels()) for axis in score_axes)
    assert [axis.get_ylabel() for axis in objective_axes] == [
        "",
        r"$J_{\mathrm{mol}}$",
        "",
        r"$J_{\mathrm{mol}}$",
    ]
    assert all(any(label.get_visible() for label in axis.get_yticklabels()) for axis in objective_axes)
    u_axes = list(figure._summary_axes[2::4])
    v_axes = list(figure._summary_axes[3::4])
    assert [axis.get_ylabel() for axis in u_axes] == [r"$u$", "", r"$u$", ""]
    assert [axis.get_ylabel() for axis in v_axes] == ["", r"$\nu$", "", r"$\nu$"]
    assert all(any(label.get_visible() for label in axis.get_yticklabels()) for axis in u_axes)
    assert all(any(label.get_visible() for label in axis.get_yticklabels()) for axis in v_axes)
    assert all(axis.get_xticks().size == 0 for axis in (*u_axes, *v_axes))
    assert len(figure._summary_colourbars) == 4
    assert len(figure._summary_row_labels) == 2
    assert len(figure._summary_column_labels) == 2


def test_signed_log_background_sweep_still_uses_categorical_distribution_axis():
    values = (-1.0, -0.1, -0.01, 0.01, 0.1, 1.0)
    runs = [
        sample_run(index, N=4, r_bg=value, score_shift=0.1 * index)
        for index, value in enumerate(values, start=1)
    ]

    figures = plot_standard_figures(runs)

    figure, axes = figures["sweep_summary"]
    assert figure._sweep_parameter == "r_bg"
    assert len(figure._summary_records) == len(values)
    assert len(axes) == 4 * len(values)
    assert [record["sweep_value"] for record in figure._summary_records] == list(values)


def test_multiple_parameter_sweeps_require_explicit_selection():
    runs = [
        sample_run(1, N=4, u_max=1.0),
        sample_run(2, N=8, u_max=10.0, score_shift=0.2),
    ]

    with pytest.raises(ValueError, match="vary multiple configuration parameters"):
        plot_standard_figures(runs)

    figures = plot_standard_figures(runs, sweep_parameter="u_max")
    assert figures["sweep_summary"][0]._sweep_parameter == "u_max"


def test_standard_figures_save_as_one_unified_summary(tmp_path):
    saved = save_standard_figures(
        [sample_run(1, N=4)],
        tmp_path,
        formats=("png", "pdf"),
    )

    assert {path.name for formats in saved.values() for path in formats.values()} == {
        "01_sweep_summary.png",
        "01_sweep_summary.pdf",
    }
    assert all(path.is_file() for formats in saved.values() for path in formats.values())
