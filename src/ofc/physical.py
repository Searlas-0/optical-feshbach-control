"""Pure conversions between laboratory and dimensionless OFC parameters.

The numerical model uses

``r_bg = a_bg * sqrt(m / (hbar * t_star))``,
``tau = T / t_star``, ``u_max = Gamma_max / gamma``, and
``v_max = nu_max / gamma``.

Each ``solve_*`` function accepts exactly one unknown, represented by ``None``,
and returns that value. Inputs may be finite scalars or two-element ranges. A
calculation involving a range returns the enclosing ``(lower, upper)`` tuple.
The functions have no config, storage, or JAX dependencies, so the returned
values can be used by either configuration creation or result queries.

``AtomConfiguration`` fixes atom, gas, linewidth, and short-time scales once
and exposes the same conversions as methods. ``molecular_density`` restores the
dimensional pair-density factor omitted from the optimized yield objective.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import TypeAlias


# h is exact in SI units; deriving hbar avoids tying this pure module to SciPy.
PLANCK_CONSTANT = 6.626_070_15e-34
HBAR = PLANCK_CONSTANT / (2.0 * math.pi)

ScalarOrRange: TypeAlias = float | tuple[float, float]

__all__ = [
    "AtomConfiguration",
    "HBAR",
    "PLANCK_CONSTANT",
    "ScalarOrRange",
    "molecular_density",
    "physical_to_dimensionless",
    "solve_background_scale",
    "solve_detuning_scale",
    "solve_optical_width",
    "solve_time_scale",
]


@dataclass(frozen=True)
class _Interval:
    lower: float
    upper: float
    is_range: bool = False

    def result(self) -> ScalarOrRange:
        if self.is_range:
            return (self.lower, self.upper)
        return self.lower


def _as_interval(name: str, value) -> _Interval:
    """Normalize one scalar or two-endpoint range without requiring NumPy."""

    if not isinstance(value, (str, bytes)):
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            converted = tolist()
            if converted is not value:
                value = converted

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"{name} range must contain exactly two endpoints.")
        endpoints = tuple(value)
        is_range = True
    else:
        endpoints = (value, value)
        is_range = False

    normalized = []
    for endpoint in endpoints:
        if isinstance(endpoint, bool) or not isinstance(endpoint, Real):
            raise ValueError(f"{name} must be a finite number or two-number range.")
        endpoint = float(endpoint)
        if not math.isfinite(endpoint):
            raise ValueError(f"{name} must be finite.")
        normalized.append(endpoint)

    lower, upper = normalized
    if lower > upper:
        raise ValueError(f"{name} range endpoints must be in increasing order.")
    return _Interval(lower, upper, is_range)


def _positive(name: str, value) -> _Interval:
    interval = _as_interval(name, value)
    if interval.lower <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return interval


def _nonnegative(name: str, value) -> _Interval:
    interval = _as_interval(name, value)
    if interval.lower < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return interval


def _signed_nonzero(name: str, value) -> _Interval:
    interval = _as_interval(name, value)
    if interval.lower <= 0.0 <= interval.upper:
        raise ValueError(f"{name} must be non-zero and its range cannot cross zero.")
    return interval


def _multiply(left: _Interval, right: _Interval) -> _Interval:
    products = (
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper,
    )
    return _Interval(min(products), max(products), left.is_range or right.is_range)


def _divide(numerator: _Interval, denominator: _Interval) -> _Interval:
    if denominator.lower <= 0.0 <= denominator.upper:
        raise ValueError("Cannot divide by a range containing zero.")
    reciprocals = _Interval(
        min(1.0 / denominator.lower, 1.0 / denominator.upper),
        max(1.0 / denominator.lower, 1.0 / denominator.upper),
        denominator.is_range,
    )
    return _multiply(numerator, reciprocals)


def _square(value: _Interval) -> _Interval:
    maximum = max(value.lower**2, value.upper**2)
    minimum = (
        0.0
        if value.lower <= 0.0 <= value.upper
        else min(value.lower**2, value.upper**2)
    )
    return _Interval(minimum, maximum, value.is_range)


def _sqrt(value: _Interval) -> _Interval:
    if value.lower < 0.0:
        raise ValueError("Cannot take the square root of a negative range.")
    return _Interval(math.sqrt(value.lower), math.sqrt(value.upper), value.is_range)


def _unknown_name(values: dict[str, object]) -> str:
    missing = [name for name, value in values.items() if value is None]
    if len(missing) != 1:
        names = ", ".join(values)
        raise ValueError(
            f"Exactly one of {names} must be None; received {len(missing)}."
        )
    return missing[0]


def _positive_hbar(hbar) -> _Interval:
    value = _positive("hbar", hbar)
    if value.is_range:
        raise ValueError("hbar must be a scalar.")
    return value


def solve_background_scale(
    *,
    a_bg=None,
    m=None,
    t_star=None,
    r_bg=None,
    hbar=HBAR,
) -> ScalarOrRange:
    """Solve one unknown in ``r_bg = a_bg sqrt(m / (hbar t_star))``.

    With the default ``hbar``, use metres for ``a_bg``, kilograms for ``m``,
    and seconds for ``t_star``. ``a_bg`` and ``r_bg`` are signed and must not
    cross zero; mass and time are positive.
    """

    supplied = {"a_bg": a_bg, "m": m, "t_star": t_star, "r_bg": r_bg}
    unknown = _unknown_name(supplied)
    values = {
        name: (
            None
            if value is None
            else _signed_nonzero(name, value)
            if name in {"a_bg", "r_bg"}
            else _positive(name, value)
        )
        for name, value in supplied.items()
    }
    hbar_value = _positive_hbar(hbar)

    if values["a_bg"] is not None and values["r_bg"] is not None:
        a_interval = values["a_bg"]
        r_interval = values["r_bg"]
        same_sign = (a_interval.lower > 0.0 and r_interval.lower > 0.0) or (
            a_interval.upper < 0.0 and r_interval.upper < 0.0
        )
        if not same_sign:
            raise ValueError("a_bg and r_bg must have the same sign.")

    if unknown == "r_bg":
        scale = _sqrt(
            _divide(values["m"], _multiply(hbar_value, values["t_star"]))
        )
        result = _multiply(values["a_bg"], scale)
    elif unknown == "a_bg":
        scale = _sqrt(
            _divide(_multiply(hbar_value, values["t_star"]), values["m"])
        )
        result = _multiply(values["r_bg"], scale)
    elif unknown == "m":
        ratio = _divide(values["r_bg"], values["a_bg"])
        result = _multiply(
            _multiply(hbar_value, values["t_star"]), _square(ratio)
        )
    else:
        ratio = _divide(values["a_bg"], values["r_bg"])
        result = _multiply(_divide(values["m"], hbar_value), _square(ratio))
    return result.result()


def solve_time_scale(*, T=None, t_star=None, tau=None) -> ScalarOrRange:
    """Solve one unknown in ``tau = T / t_star``.

    ``tau`` is stored as ``t_interval`` by the optimization configuration.
    """

    supplied = {"T": T, "t_star": t_star, "tau": tau}
    unknown = _unknown_name(supplied)
    values = {
        name: None if value is None else _positive(name, value)
        for name, value in supplied.items()
    }
    if unknown == "tau":
        result = _divide(values["T"], values["t_star"])
    elif unknown == "T":
        result = _multiply(values["tau"], values["t_star"])
    else:
        result = _divide(values["T"], values["tau"])
    return result.result()


def _solve_positive_ratio(
    *,
    numerator_name: str,
    denominator_name: str,
    ratio_name: str,
    numerator,
    denominator,
    ratio,
) -> ScalarOrRange:
    supplied = {
        numerator_name: numerator,
        denominator_name: denominator,
        ratio_name: ratio,
    }
    unknown = _unknown_name(supplied)
    values = {
        name: None if value is None else _positive(name, value)
        for name, value in supplied.items()
    }
    if unknown == ratio_name:
        result = _divide(values[numerator_name], values[denominator_name])
    elif unknown == numerator_name:
        result = _multiply(values[ratio_name], values[denominator_name])
    else:
        result = _divide(values[numerator_name], values[ratio_name])
    return result.result()


def solve_optical_width(
    *, gamma=None, Gamma_max=None, u_max=None
) -> ScalarOrRange:
    """Solve one unknown in ``u_max = Gamma_max / gamma``.

    ``gamma`` and ``Gamma_max`` may use Hz, rad/s, or another rate unit, but
    they must use the same unit.
    """

    return _solve_positive_ratio(
        numerator_name="Gamma_max",
        denominator_name="gamma",
        ratio_name="u_max",
        numerator=Gamma_max,
        denominator=gamma,
        ratio=u_max,
    )


def solve_detuning_scale(
    *, gamma=None, nu_max=None, v_max=None
) -> ScalarOrRange:
    """Solve one unknown in ``v_max = nu_max / gamma``.

    ``nu_max`` is the positive magnitude of the signed detuning bound. It must
    use the same frequency/rate convention as ``gamma``.
    """

    return _solve_positive_ratio(
        numerator_name="nu_max",
        denominator_name="gamma",
        ratio_name="v_max",
        numerator=nu_max,
        denominator=gamma,
        ratio=v_max,
    )


def physical_to_dimensionless(
    *,
    a_bg,
    m,
    t_star,
    T,
    gamma,
    Gamma_max,
    nu_max,
    hbar=HBAR,
) -> dict[str, ScalarOrRange]:
    """Return the four physical parameters consumed by an OFC calculation.

    The output names match :class:`ofc.config.ResolvedConfig`. Scalar physical
    inputs produce scalar values. If any relevant input is a range, that output
    is an enclosing ``(lower, upper)`` tuple, which also matches the range
    convention used by :meth:`ofc.results.Results.search`.
    """

    return {
        "r_bg": solve_background_scale(
            a_bg=a_bg, m=m, t_star=t_star, r_bg=None, hbar=hbar
        ),
        "t_interval": solve_time_scale(T=T, t_star=t_star, tau=None),
        "u_max": solve_optical_width(
            gamma=gamma, Gamma_max=Gamma_max, u_max=None
        ),
        "v_max": solve_detuning_scale(gamma=gamma, nu_max=nu_max, v_max=None),
    }


def molecular_density(
    dimensionless_yield,
    *,
    g_2,
    l_star,
) -> ScalarOrRange:
    """Convert the optimized yield objective to a physical product density.

    Under the loss-rate normalization implemented by :mod:`ofc.physics`,

    ``n_mol = g_2(0) * l_star**3 * dimensionless_yield``.

    ``g_2`` is the *unnormalized* initial equal-position pair density, with
    units of inverse length to the sixth power. It is not the commonly used
    dimensionless normalized coherence ``g^(2)(0)``. If ``l_star`` is in
    metres and ``g_2`` in ``m^-6``, the result is in ``m^-3``.

    This conversion follows the code's current ``1/(2 pi)`` loss prefactor.
    Whether that prefactor represents product molecules, lost pairs, or
    depleted atoms remains a separate counting-convention question in the
    physical model.
    """

    yield_value = _nonnegative("dimensionless_yield", dimensionless_yield)
    pair_density = _nonnegative("g_2", g_2)
    length = _positive("l_star", l_star)
    length_cubed = _multiply(_multiply(length, length), length)
    return _multiply(_multiply(pair_density, length_cubed), yield_value).result()


def _scalar(interval: _Interval, name: str) -> float:
    if interval.is_range:
        raise ValueError(f"{name} must be a scalar for an AtomConfiguration.")
    return interval.lower


@dataclass(frozen=True)
class AtomConfiguration:
    """Fixed physical scales for one atom, gas state, and optical transition.

    ``a_bg`` is in metres, ``m`` in kilograms, ``g_2`` in ``m^-6``, and
    ``t_star`` in seconds when the default SI ``hbar`` is used. ``gamma`` may
    use Hz or angular frequency, provided all optical rates passed to this
    object use the same convention.

    The configured ``t_star`` is also treated as the inclusive upper bound of
    the requested short-time frame. Object time conversions reject a physical
    duration above it. If no duration is supplied, they use ``T=t_star``.
    """

    a_bg: float
    gamma: float
    g_2: float
    m: float
    t_star: float
    hbar: float = HBAR

    def __post_init__(self) -> None:
        values = {
            "a_bg": _scalar(_signed_nonzero("a_bg", self.a_bg), "a_bg"),
            "gamma": _scalar(_positive("gamma", self.gamma), "gamma"),
            "g_2": _scalar(_nonnegative("g_2", self.g_2), "g_2"),
            "m": _scalar(_positive("m", self.m), "m"),
            "t_star": _scalar(_positive("t_star", self.t_star), "t_star"),
            "hbar": _scalar(_positive("hbar", self.hbar), "hbar"),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def l_star(self) -> float:
        """Reference length ``sqrt(hbar * t_star / m)``."""

        return math.sqrt(self.hbar * self.t_star / self.m)

    @property
    def short_time_interval(self) -> float:
        """Inclusive physical-time limit used by object time conversions."""

        return self.t_star

    @property
    def r_bg(self) -> float:
        """Dimensionless background scattering length ``a_bg / l_star``."""

        return self.a_bg / self.l_star

    def solve_background_scale(self) -> float:
        """Return the fixed configuration's dimensionless ``r_bg``."""

        return self.r_bg

    def _validate_physical_time(self, T) -> None:
        interval = _positive("T", T)
        if interval.upper > self.t_star and not math.isclose(
            interval.upper, self.t_star, rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError(
                f"T={interval.result()} exceeds the configured short-time "
                f"interval t_star={self.t_star}."
            )

    def solve_time_scale(self, *, T=None, tau=None) -> ScalarOrRange:
        """Solve ``tau=T/t_star``, defaulting an empty call to ``T=t_star``.

        A supplied or calculated physical ``T`` must lie inside this object's
        short-time frame.
        """

        if T is None and tau is None:
            T = self.t_star
        if T is not None:
            self._validate_physical_time(T)
        result = solve_time_scale(T=T, t_star=self.t_star, tau=tau)
        if T is None:
            self._validate_physical_time(result)
        return result

    def solve_optical_width(
        self, *, Gamma_max=None, u_max=None
    ) -> ScalarOrRange:
        """Solve ``u_max=Gamma_max/gamma`` using the fixed linewidth."""

        return solve_optical_width(
            gamma=self.gamma, Gamma_max=Gamma_max, u_max=u_max
        )

    def solve_detuning_scale(
        self, *, nu_max=None, v_max=None
    ) -> ScalarOrRange:
        """Solve ``v_max=nu_max/gamma`` using the fixed linewidth."""

        return solve_detuning_scale(
            gamma=self.gamma, nu_max=nu_max, v_max=v_max
        )

    def dimensionless_parameters(
        self,
        *,
        Gamma_max,
        nu_max,
        T=None,
    ) -> dict[str, ScalarOrRange]:
        """Return calculation parameters, defaulting ``T`` to ``t_star``."""

        return {
            "r_bg": self.r_bg,
            "t_interval": self.solve_time_scale(T=T, tau=None),
            "u_max": self.solve_optical_width(Gamma_max=Gamma_max, u_max=None),
            "v_max": self.solve_detuning_scale(nu_max=nu_max, v_max=None),
        }

    def molecular_density(self, dimensionless_yield) -> ScalarOrRange:
        """Return product density using this configuration's ``g_2`` and scale."""

        return molecular_density(
            dimensionless_yield,
            g_2=self.g_2,
            l_star=self.l_star,
        )
