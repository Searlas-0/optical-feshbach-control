import numpy as np
import pytest

from ofc.config import ResolvedConfig
from ofc.physical import (
    AtomConfiguration,
    molecular_density,
    physical_to_dimensionless,
    solve_background_scale,
    solve_detuning_scale,
    solve_optical_width,
    solve_time_scale,
)


def test_background_scale_solves_each_parameter():
    known = {"a_bg": -2.0, "m": 8.0, "t_star": 2.0, "r_bg": -4.0}

    for missing, expected in known.items():
        arguments = {**known, missing: None}
        actual = solve_background_scale(**arguments, hbar=1.0)
        assert actual == pytest.approx(expected)


def test_time_scale_solves_each_parameter():
    assert solve_time_scale(T=4.0, t_star=2.0, tau=None) == pytest.approx(2.0)
    assert solve_time_scale(T=None, t_star=2.0, tau=2.0) == pytest.approx(4.0)
    assert solve_time_scale(T=4.0, t_star=None, tau=2.0) == pytest.approx(2.0)


def test_control_scales_solve_each_parameter():
    assert solve_optical_width(
        gamma=5.0, Gamma_max=20.0, u_max=None
    ) == pytest.approx(4.0)
    assert solve_optical_width(
        gamma=5.0, Gamma_max=None, u_max=4.0
    ) == pytest.approx(20.0)
    assert solve_optical_width(
        gamma=None, Gamma_max=20.0, u_max=4.0
    ) == pytest.approx(5.0)

    assert solve_detuning_scale(
        gamma=5.0, nu_max=50.0, v_max=None
    ) == pytest.approx(10.0)
    assert solve_detuning_scale(
        gamma=5.0, nu_max=None, v_max=10.0
    ) == pytest.approx(50.0)
    assert solve_detuning_scale(
        gamma=None, nu_max=50.0, v_max=10.0
    ) == pytest.approx(5.0)


def test_ranges_are_propagated_to_enclosing_ordered_ranges():
    assert solve_time_scale(
        T=[1e-6, 1e-5], t_star=1e-5, tau=None
    ) == pytest.approx((0.1, 1.0))
    assert solve_optical_width(
        gamma=[2.0, 4.0], Gamma_max=[20.0, 40.0], u_max=None
    ) == pytest.approx((5.0, 20.0))
    assert solve_background_scale(
        a_bg=[-2.0, -1.0],
        m=[8.0, 18.0],
        t_star=[2.0, 8.0],
        r_bg=None,
        hbar=1.0,
    ) == pytest.approx((-6.0, -1.0))


def test_numpy_ranges_and_scalars_are_accepted():
    actual = solve_time_scale(
        T=np.asarray([1e-6, 1e-5]),
        t_star=np.float64(1e-5),
        tau=None,
    )
    assert actual == pytest.approx((0.1, 1.0))


def test_physical_to_dimensionless_uses_current_config_names():
    values = physical_to_dimensionless(
        a_bg=-2.0,
        m=8.0,
        t_star=2.0,
        T=[1.0, 4.0],
        gamma=5.0,
        Gamma_max=20.0,
        nu_max=50.0,
        hbar=1.0,
    )

    assert values == {
        "r_bg": pytest.approx(-4.0),
        "t_interval": pytest.approx((0.5, 2.0)),
        "u_max": pytest.approx(4.0),
        "v_max": pytest.approx(10.0),
    }


def test_scalar_conversion_can_construct_a_resolved_calculation_config():
    values = physical_to_dimensionless(
        a_bg=-2.0,
        m=8.0,
        t_star=2.0,
        T=4.0,
        gamma=5.0,
        Gamma_max=20.0,
        nu_max=50.0,
        hbar=1.0,
    )

    config = ResolvedConfig(**values)
    assert config.r_bg == pytest.approx(-4.0)
    assert config.t_interval == pytest.approx(2.0)
    assert config.u_max == pytest.approx(4.0)
    assert config.v_max == pytest.approx(10.0)


def test_molecular_density_restores_pair_density_and_length_scale():
    assert molecular_density(
        5.0, g_2=3.0, l_star=2.0
    ) == pytest.approx(120.0)
    assert molecular_density(
        [1.0, 2.0], g_2=[3.0, 4.0], l_star=[2.0, 3.0]
    ) == pytest.approx((24.0, 216.0))
    assert molecular_density(0.0, g_2=3.0, l_star=2.0) == pytest.approx(0.0)


@pytest.fixture
def atom():
    return AtomConfiguration(
        a_bg=-2.0,
        gamma=5.0,
        g_2=3.0,
        m=8.0,
        t_star=2.0,
        hbar=1.0,
    )


def test_atom_configuration_exposes_fixed_scales(atom):
    assert atom.l_star == pytest.approx(0.5)
    assert atom.short_time_interval == pytest.approx(2.0)
    assert atom.r_bg == pytest.approx(-4.0)
    assert atom.solve_background_scale() == pytest.approx(-4.0)


def test_atom_configuration_uses_fixed_parameters_for_conversions(atom):
    assert atom.solve_time_scale() == pytest.approx(1.0)
    assert atom.solve_time_scale(T=[0.5, 2.0], tau=None) == pytest.approx(
        (0.25, 1.0)
    )
    assert atom.solve_time_scale(T=None, tau=0.5) == pytest.approx(1.0)
    assert atom.solve_optical_width(
        Gamma_max=20.0, u_max=None
    ) == pytest.approx(4.0)
    assert atom.solve_optical_width(
        Gamma_max=None, u_max=4.0
    ) == pytest.approx(20.0)
    assert atom.solve_detuning_scale(
        nu_max=50.0, v_max=None
    ) == pytest.approx(10.0)
    assert atom.solve_detuning_scale(
        nu_max=None, v_max=10.0
    ) == pytest.approx(50.0)


def test_atom_configuration_rejects_times_outside_short_time_frame(atom):
    with pytest.raises(ValueError, match="exceeds the configured short-time"):
        atom.solve_time_scale(T=2.1, tau=None)
    with pytest.raises(ValueError, match="exceeds the configured short-time"):
        atom.solve_time_scale(T=None, tau=1.1)


def test_atom_configuration_returns_dimensionless_calculation_data(atom):
    values = atom.dimensionless_parameters(Gamma_max=20.0, nu_max=50.0)
    assert values == {
        "r_bg": pytest.approx(-4.0),
        "t_interval": pytest.approx(1.0),
        "u_max": pytest.approx(4.0),
        "v_max": pytest.approx(10.0),
    }
    assert ResolvedConfig(**values).t_interval == pytest.approx(1.0)


def test_atom_configuration_converts_yield_with_its_fixed_pair_density(atom):
    assert atom.molecular_density(2.0) == pytest.approx(0.75)
    assert atom.molecular_density([0.0, 2.0]) == pytest.approx((0.0, 0.75))


def test_atom_configuration_requires_scalar_fixed_parameters():
    with pytest.raises(ValueError, match="must be a scalar"):
        AtomConfiguration(
            a_bg=[-2.0, -1.0],
            gamma=5.0,
            g_2=3.0,
            m=8.0,
            t_star=2.0,
            hbar=1.0,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"T": None, "t_star": None, "tau": 1.0},
        {"T": 1.0, "t_star": 1.0, "tau": 1.0},
    ],
)
def test_exactly_one_parameter_must_be_unknown(arguments):
    with pytest.raises(ValueError, match="Exactly one"):
        solve_time_scale(**arguments)


@pytest.mark.parametrize(
    ("function", "arguments", "message"),
    [
        (
            solve_time_scale,
            {"T": [2.0, 1.0], "t_star": 1.0, "tau": None},
            "in increasing order",
        ),
        (
            solve_time_scale,
            {"T": [1.0, 2.0, 3.0], "t_star": 1.0, "tau": None},
            "exactly two endpoints",
        ),
        (
            solve_optical_width,
            {"gamma": 0.0, "Gamma_max": 2.0, "u_max": None},
            "gamma must be positive",
        ),
        (
            solve_background_scale,
            {"a_bg": [-1.0, 1.0], "m": 1.0, "t_star": 1.0, "r_bg": None},
            "cannot cross zero",
        ),
        (
            solve_background_scale,
            {"a_bg": -1.0, "m": None, "t_star": 1.0, "r_bg": 1.0},
            "same sign",
        ),
        (
            molecular_density,
            {"dimensionless_yield": -1.0, "g_2": 1.0, "l_star": 1.0},
            "dimensionless_yield must be non-negative",
        ),
    ],
)
def test_invalid_physical_values_fail_clearly(function, arguments, message):
    with pytest.raises(ValueError, match=message):
        function(**arguments)
