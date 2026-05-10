"""Small-bT TMDPDF matching kernels for collinear PDFs.

The N3LO unpolarized and helicity kernels are evaluated from the numerical
ancillary expressions distributed with arXiv:2012.03256 and arXiv:2509.01655.
The transversity logarithmic structure follows arXiv:2509.17568, with the
transversity splitting kernels read from its ``tPS["qq"]`` ancillary expression.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from scipy.integrate import quad
from scipy.special import factorial, spence, zeta

from lametlat.utils.constants import CA, CF, TF, Lz_func, alphas_nloop, beta


_ORDER_TO_POWER = {"NLO": 1, "NNLO": 2, "N3LO": 3}
_POLY_POINTS = (0.0, 1.0, -1.0, 2.0)
_POLY_VANDERMONDE_INV = np.linalg.inv(np.vander(_POLY_POINTS, 4, increasing=True))


@dataclass(frozen=True)
class _Distribution:
    regular: float = 0.0
    delta: float = 0.0
    plus: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def _is_regular_only(self) -> bool:
        return self.delta == 0.0 and all(val == 0.0 for val in self.plus)

    def __add__(self, other):
        other = _to_distribution(other)
        return _Distribution(
            self.regular + other.regular,
            self.delta + other.delta,
            tuple(a + b for a, b in zip(self.plus, other.plus)),
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-_to_distribution(other))

    def __rsub__(self, other):
        return _to_distribution(other) + (-self)

    def __neg__(self):
        return _Distribution(-self.regular, -self.delta, tuple(-val for val in self.plus))

    def __mul__(self, other):
        other = _to_distribution(other)
        if self._is_regular_only():
            return _scale_distribution(other, self.regular)
        if other._is_regular_only():
            return _scale_distribution(self, other.regular)
        raise ValueError("Products of distributions are not supported.")

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _to_distribution(other)
        if not other._is_regular_only():
            raise ValueError("Division by a distribution is not supported.")
        return _scale_distribution(self, 1.0 / other.regular)

    def __rtruediv__(self, other):
        if not self._is_regular_only():
            raise ValueError("Division by a distribution is not supported.")
        return _to_distribution(other) / self.regular

    def __pow__(self, power):
        if not self._is_regular_only():
            raise ValueError("Powers of distributions are not supported.")
        return _Distribution(self.regular**power)


def _to_distribution(val) -> _Distribution:
    if isinstance(val, _Distribution):
        return val
    return _Distribution(float(val))


def _scale_distribution(dist: _Distribution, factor: float) -> _Distribution:
    return _Distribution(
        factor * dist.regular,
        factor * dist.delta,
        tuple(factor * val for val in dist.plus),
    )


_DELTA = _Distribution(delta=1.0)
_PLUS = {
    0: _Distribution(plus=(1.0, 0.0, 0.0)),
    1: _Distribution(plus=(0.0, 1.0, 0.0)),
    2: _Distribution(plus=(0.0, 0.0, 1.0)),
}


def _safe_z(z: float) -> float:
    return float(np.clip(z, 1e-12, 1.0 - 1e-10))


def _max_power(order: str) -> int:
    return _ORDER_TO_POWER[order.upper()]


def _identity_kernel(x_ls_mat, y_ls_mat):
    identity = np.zeros([len(x_ls_mat), len(y_ls_mat)])
    for idx in range(min(len(x_ls_mat), len(y_ls_mat))):
        identity[idx][idx] = 1.0
    return identity


def _check_identical_grids(x_ls_mat, y_ls_mat) -> None:
    if len(x_ls_mat) != len(y_ls_mat) or not np.allclose(x_ls_mat, y_ls_mat):
        raise ValueError("small-bT matching kernels currently assume identical x/y grids.")


def _build_plus_kernel_matrix(x_ls_mat, y_ls_mat, sing_piece_func):
    _check_identical_grids(x_ls_mat, y_ls_mat)
    dy = abs(y_ls_mat[1] - y_ls_mat[0])
    plus_kernel = np.zeros([len(x_ls_mat), len(y_ls_mat)])

    for idx, x_val in enumerate(x_ls_mat):
        for idy in range(idx + 1, len(y_ls_mat)):
            y_val = y_ls_mat[idy]
            plus_kernel[idx][idy] = dy / y_val * sing_piece_func(x_val / y_val)

    for idy in range(len(y_ls_mat)):
        plus_kernel[idy][idy] = -np.sum(plus_kernel[:, idy])
    return plus_kernel


def _build_regular_kernel_matrix(x_ls_mat, y_ls_mat, regular_piece_func, *, integrate_diagonal=True):
    _check_identical_grids(x_ls_mat, y_ls_mat)
    dy = abs(y_ls_mat[1] - y_ls_mat[0])
    regular_kernel = np.zeros([len(x_ls_mat), len(y_ls_mat)])

    def safe_piece(z):
        return regular_piece_func(_safe_z(z))

    for idx, x_val in enumerate(x_ls_mat):
        upper = y_ls_mat[idx + 1] if idx + 1 < len(y_ls_mat) else 1.0
        if upper > x_val:
            if integrate_diagonal:
                regular_kernel[idx][idx] = quad(
                    lambda y: safe_piece(x_val / y) / y,
                    x_val,
                    upper,
                    points=[x_val],
                    epsabs=1e-10,
                    epsrel=1e-10,
                    limit=200,
                )[0]
            else:
                y_mid = (x_val + upper) / 2.0
                regular_kernel[idx][idx] = dy / y_mid * safe_piece(x_val / y_mid)

        for idy in range(idx + 1, len(y_ls_mat)):
            y_val = y_ls_mat[idy]
            regular_kernel[idx][idy] = dy / y_val * safe_piece(x_val / y_val)
    return regular_kernel


def _build_plus_kernel_rows(x_ls_mat, y_ls_mat, row_indices, sing_piece_func):
    _check_identical_grids(x_ls_mat, y_ls_mat)
    dy = abs(y_ls_mat[1] - y_ls_mat[0])
    rows = np.zeros([len(row_indices), len(y_ls_mat)])

    for row_idx, idx in enumerate(row_indices):
        x_val = x_ls_mat[idx]
        for idy in range(idx + 1, len(y_ls_mat)):
            y_val = y_ls_mat[idy]
            rows[row_idx][idy] = dy / y_val * sing_piece_func(x_val / y_val)
        y_val = y_ls_mat[idx]
        rows[row_idx][idx] = -sum(
            dy / y_val * sing_piece_func(x_ls_mat[col_idx] / y_val)
            for col_idx in range(idx)
        )
    return rows


def _build_regular_kernel_rows(x_ls_mat, y_ls_mat, row_indices, regular_piece_func, *, integrate_diagonal=True):
    _check_identical_grids(x_ls_mat, y_ls_mat)
    dy = abs(y_ls_mat[1] - y_ls_mat[0])
    rows = np.zeros([len(row_indices), len(y_ls_mat)])

    def safe_piece(z):
        return regular_piece_func(_safe_z(z))

    for row_idx, idx in enumerate(row_indices):
        x_val = x_ls_mat[idx]
        upper = y_ls_mat[idx + 1] if idx + 1 < len(y_ls_mat) else 1.0
        if upper > x_val:
            if integrate_diagonal:
                rows[row_idx][idx] = quad(
                    lambda y: safe_piece(x_val / y) / y,
                    x_val,
                    upper,
                    points=[x_val],
                    epsabs=1e-10,
                    epsrel=1e-10,
                    limit=200,
                )[0]
            else:
                y_mid = (x_val + upper) / 2.0
                rows[row_idx][idx] = dy / y_mid * safe_piece(x_val / y_mid)
        for idy in range(idx + 1, len(y_ls_mat)):
            y_val = y_ls_mat[idy]
            rows[row_idx][idy] = dy / y_val * safe_piece(x_val / y_val)
    return rows


def _mathematica_to_python_expr(expr: str) -> str:
    expr = re.sub(r"(\d+(?:\.\d*)?|\.\d+)\*\^([+-]?\d+)", r"\1e\2", expr)
    expr = re.sub(r"plusD\[(\d+)(?:\.)?,\s*1\s*-\s*x\]", r"PLUS\1", expr)
    expr = re.sub(r"delta\[\s*1\s*-\s*x\s*\]", "DELTA", expr)
    expr = re.sub(r"\bas\b", "a_s", expr)
    expr = expr.replace("\n", " ")
    expr = expr.replace("Log[", "log(")
    expr = expr.replace("zeta[", "zeta(")
    expr = expr.replace("Zeta[", "zeta(")
    expr = expr.replace("H[", "H(")
    expr = expr.replace("]", ")")
    expr = expr.replace("^", "**")
    return expr


def _load_assignment_expr(path: Path, family: str, channel: str) -> str:
    text = path.read_text()
    marker = f'{family}["{channel}"]'
    start = text.index(marker)
    rhs_start = text.index("=", start) + 1
    next_match = re.search(rf"\n{re.escape(family)}\[\"", text[rhs_start:])
    if next_match is None:
        return text[rhs_start:].strip()
    return text[rhs_start : rhs_start + next_match.start()].strip()


@lru_cache(maxsize=None)
def _load_distribution_code(path_str: str, family: str, channel: str):
    expr = _mathematica_to_python_expr(_load_assignment_expr(Path(path_str), family, channel))
    return compile(f"({expr})", f"{path_str}:{family}[{channel}]", "eval")


_HPL_WORD_CACHE: dict[str, tuple[tuple[int, ...], ...]] = {}
_HPL_TABLE_CACHE: dict[tuple[str, float], dict[tuple[int, ...], float]] = {}


def _hpl_expand_index(idx: int) -> tuple[int, ...]:
    if idx in (-1, 0, 1):
        return (idx,)
    if idx > 1:
        return (0,) * (idx - 1) + (1,)
    return (0,) * (-idx - 1) + (-1,)


def _hpl_expand(indices) -> tuple[int, ...]:
    expanded: tuple[int, ...] = ()
    for idx in indices:
        expanded += _hpl_expand_index(int(idx))
    return expanded


def _hpl_words_for_path(path: Path) -> tuple[tuple[int, ...], ...]:
    path_str = str(path)
    if path_str in _HPL_WORD_CACHE:
        return _HPL_WORD_CACHE[path_str]
    text = path.read_text()
    words = {()}
    for match in re.finditer(r"H\[([^\]]+), x\]", text):
        raw_word = tuple(int(part.strip()) for part in match.group(1).split(","))
        word = _hpl_expand(raw_word)
        for pos in range(len(word)):
            words.add(word[pos:])
        words.add(word)
    _HPL_WORD_CACHE[path_str] = tuple(sorted(words, key=len))
    return _HPL_WORD_CACHE[path_str]


def _hpl_fast_table(path: Path, x: float) -> dict[tuple[int, ...], float]:
    x = float(np.clip(x, 1e-12, 1.0 - 1e-10))
    key = (str(path), round(x, 12))
    if key in _HPL_TABLE_CACHE:
        return _HPL_TABLE_CACHE[key]

    n_grid = 2000
    grid = np.linspace(1e-10, x, n_grid)
    table = {(): np.ones_like(grid)}

    for word in _hpl_words_for_path(path):
        if word == ():
            continue
        if all(idx == 0 for idx in word):
            table[word] = np.log(grid) ** len(word) / factorial(len(word))
            continue

        head = word[0]
        tail_vals = table[word[1:]]
        if head == 0:
            integrand = tail_vals / grid
        elif head == 1:
            integrand = tail_vals / (1.0 - grid)
        elif head == -1:
            integrand = tail_vals / (1.0 + grid)
        else:
            raise ValueError(f"Invalid HPL index: {head}")

        vals = np.zeros_like(grid)
        vals[1:] = np.cumsum((integrand[1:] + integrand[:-1]) * np.diff(grid) / 2.0)
        table[word] = vals

    reduced = {word: vals[-1] for word, vals in table.items()}
    _HPL_TABLE_CACHE[key] = reduced
    return reduced


def _eval_distribution_code(code, path: Path, x: float, a_s: float, Lp: float, Lh: float, Nf: int):
    def H(*args):
        return _hpl_fast_table(path, args[-1])[_hpl_expand(args[:-1])]

    nc = 3.0
    return _to_distribution(eval(
        code,
        {"__builtins__": {}},
        {
            "x": x,
            "a_s": a_s,
            "Lp": Lp,
            "Lh": Lh,
            "Nf": float(Nf),
            "nc": nc,
            "dabc2": (nc**2 - 1.0) * (nc**2 - 4.0) / nc,
            "CF": CF,
            "CA": CA,
            "TF": TF,
            "DELTA": _DELTA,
            "PLUS0": _PLUS[0],
            "PLUS1": _PLUS[1],
            "PLUS2": _PLUS[2],
            "log": np.log,
            "zeta": zeta,
            "H": H,
        },
    ))


@lru_cache(maxsize=300000)
def _distribution_coeff_cached(path_str: str, family: str, channel: str, x: float, Lp: float, Lh: float, Nf: int, power: int):
    path = Path(path_str)
    code = _load_distribution_code(path_str, family, channel)
    values = [
        _eval_distribution_code(code, path, x, a_s, Lp, Lh, Nf)
        for a_s in _POLY_POINTS
    ]
    regular = _POLY_VANDERMONDE_INV[power] @ np.array([val.regular for val in values], dtype=float)
    delta = _POLY_VANDERMONDE_INV[power] @ np.array([val.delta for val in values], dtype=float)
    plus = []
    for plus_idx in range(3):
        plus.append(_POLY_VANDERMONDE_INV[power] @ np.array([val.plus[plus_idx] for val in values], dtype=float))
    return _Distribution(float(regular), float(delta), tuple(float(val) for val in plus))


def _distribution_coeff(path: Path, family: str, channel: str, x: float, Lp: float, Lh: float, Nf: int, power: int):
    return _distribution_coeff_cached(
        str(path),
        family,
        channel,
        round(_safe_z(x), 12),
        round(float(Lp), 12),
        round(float(Lh), 12),
        int(Nf),
        int(power),
    )


def _reference_root() -> Path:
    return Path(__file__).resolve().parent / "reference"


def _reference_file(filename: str) -> Path:
    return _reference_root() / filename


def _channel_distribution_coeff(kind: str, x: float, Lp: float, Lh: float, Nf: int, power: int):
    if kind == "unpolarized":
        path = _reference_file("2012.03256_TMDPDFN.m")
        return (
            _distribution_coeff(path, "TMDpdfN", "qq", x, Lp, Lh, Nf, power)
            - _distribution_coeff(path, "TMDpdfN", "qqp", x, Lp, Lh, Nf, power)
        )
    if kind == "helicity":
        path = _reference_file("2509.01655_dPDFMSbN.m")
        return _distribution_coeff(path, "dPDFMSbN", "+", x, Lp, Lh, Nf, power)
    raise ValueError(f"Unsupported ancillary channel kind: {kind}")


def _ancillary_matching_kernel_rows_by_order(
    kind: str,
    x_ls_mat,
    y_ls_mat,
    row_indices,
    b_fm: float,
    mu: float,
    zeta_scale: float,
    Nf: int,
    orders,
    integrate_diagonal=True,
):
    _check_identical_grids(x_ls_mat, y_ls_mat)
    orders = [order.upper() for order in orders]
    max_power = max(_max_power(order) for order in orders)
    requested_by_power = {_max_power(order): order for order in orders}
    a_s = alphas_nloop(mu, order=2, Nf=Nf) / (4.0 * np.pi)
    Lp = Lz_func(b_fm, mu)
    Lh = np.log(zeta_scale / mu**2)

    cumulative = np.zeros([len(row_indices), len(y_ls_mat)])
    rows_by_order = {}
    for power in range(max_power + 1):
        def dist(z, power=power):
            return _channel_distribution_coeff(kind, z, Lp, Lh, Nf, power)

        power_rows = np.zeros_like(cumulative)
        delta_coeff = dist(0.5).delta
        for row_idx, idx in enumerate(row_indices):
            power_rows[row_idx][idx] += delta_coeff
        power_rows += _build_regular_kernel_rows(
            x_ls_mat,
            y_ls_mat,
            row_indices,
            lambda z: dist(z).regular,
            integrate_diagonal=integrate_diagonal,
        )
        for plus_order in range(3):
            power_rows += _build_plus_kernel_rows(
                x_ls_mat,
                y_ls_mat,
                row_indices,
                lambda z, plus_order=plus_order: dist(z).plus[plus_order]
                * np.log1p(-_safe_z(z)) ** plus_order
                / (1.0 - _safe_z(z)),
            )

        cumulative = cumulative + a_s**power * power_rows
        if power in requested_by_power:
            rows_by_order[requested_by_power[power]] = cumulative.copy()
    return rows_by_order


def unpolarized_matching_kernel_rows_by_order(
    x_ls_mat,
    y_ls_mat,
    row_indices,
    b_fm,
    mu=2.0,
    zeta=4.0,
    Nf=3,
    orders=("NLO", "NNLO", "N3LO"),
    isov=True,
    integrate_diagonal=True,
):
    if not isov:
        raise NotImplementedError("Only the isoV nonsinglet unpolarized kernel is implemented.")
    return _ancillary_matching_kernel_rows_by_order(
        "unpolarized", x_ls_mat, y_ls_mat, row_indices, b_fm, mu, zeta, Nf, orders, integrate_diagonal
    )


def helicity_matching_kernel_rows_by_order(
    x_ls_mat,
    y_ls_mat,
    row_indices,
    b_fm,
    mu=2.0,
    zeta=4.0,
    Nf=3,
    orders=("NLO", "NNLO", "N3LO"),
    isov=True,
    integrate_diagonal=True,
):
    if not isov:
        raise NotImplementedError("Only the isoV nonsinglet helicity kernel is implemented.")
    return _ancillary_matching_kernel_rows_by_order(
        "helicity", x_ls_mat, y_ls_mat, row_indices, b_fm, mu, zeta, Nf, orders, integrate_diagonal
    )


def _li2(x: float) -> float:
    return spence(1.0 - x)


_LI3_CACHE: dict[float, float] = {}


def _li3(x: float) -> float:
    x = float(x)
    if x == 0.0:
        return 0.0
    if x in _LI3_CACHE:
        return _LI3_CACHE[x]

    def integrand(t):
        if t == 0.0:
            return 1.0
        return _li2(t) / t

    val = quad(integrand, 0.0, x, epsabs=1e-11, epsrel=1e-11, limit=200)[0]
    _LI3_CACHE[x] = val
    return val


def _trans_tps_path() -> Path:
    return _reference_file("2509.17568_tPS.m")


def _trans_splitting_distribution(power: int, x: float, Nf: int):
    return _distribution_coeff(_trans_tps_path(), "tPS", "qq", x, 0.0, 0.0, Nf, power)


def _splitting_kernel_matrix(x_ls_mat, y_ls_mat, power: int, Nf: int):
    identity = _identity_kernel(x_ls_mat, y_ls_mat)

    def dist(z):
        return _trans_splitting_distribution(power, z, Nf)

    plus = np.zeros_like(identity)
    for plus_order in range(3):
        plus += _build_plus_kernel_matrix(
            x_ls_mat,
            y_ls_mat,
            lambda z, plus_order=plus_order: dist(z).plus[plus_order] * np.log1p(-_safe_z(z)) ** plus_order / (1.0 - _safe_z(z)),
        )
    regular = _build_regular_kernel_matrix(
        x_ls_mat,
        y_ls_mat,
        lambda z: dist(z).regular,
        integrate_diagonal=False,
    )
    return plus + regular + dist(0.5).delta * identity


def _i2_qq_plus_fit_coeff(Nf=3):
    return 14.9267 + 5.53086 * Nf


def _i2_qq_regular_piece_fit(x, Nf=3):
    x = _safe_z(x)
    xbar = 1.0 - x
    lx = np.log(x)
    lxb = np.log(xbar)
    n1 = (
        x**3 * (0.159353 * lx**2 + 1.83868 * lx - 3.46249)
        + x**2 * (0.918768 * lx**2 + 3.30005 * lx + 1.07696)
        + x * (0.888889 * lx**2 + 2.96296 * lx + 2.37017)
        - 0.0100775 * x**6
        + 0.090776 * x**5
        - 0.657935 * x**4
        + 1.48148 * xbar
        - 7.90123
    )
    n0 = (
        x**3 * (2.43688 * lx**3 - 17.6078 * lx**2 + 43.7086 * lx - 39.0133)
        + x**2 * (-2.54756 * lx**3 - 13.8299 * lx**2 - 7.50098 * lx + 63.2888)
        + x * (-2.66667 * lx**3 - 9.33333 * lx**2 - 39.1111 * lx - 16.4544)
        + xbar**3 * (0.508876 * lxb**2 - 0.601064 * lxb)
        + xbar**2 * (1.19128 * lxb**2 - 2.40075 * lxb)
        - 7.11111 * lxb**2
        + 22.2222 * lxb
        + xbar * (3.55556 * lxb**2 - 13.3333 * lxb + 10.0354)
        + 0.0762313 * x**6
        - 0.683931 * x**5
        + 3.19561 * x**4
        - 9.851
    )
    return n0 + Nf * n1


def _i2_scale_independent_matrix(x_ls_mat, y_ls_mat, Nf=3):
    plus = _build_plus_kernel_matrix(x_ls_mat, y_ls_mat, lambda z: _i2_qq_plus_fit_coeff(Nf=Nf) / (1.0 - z))
    regular = _build_regular_kernel_matrix(x_ls_mat, y_ls_mat, lambda z: _i2_qq_regular_piece_fit(z, Nf=Nf))
    return plus + regular


def _i3_qq_plus_fit_coeff(Nf=3):
    return 140.136 + 154.257 * Nf - 9.09324 * Nf**2


def _i3_qq_regular_piece_fit(x, Nf=3):
    x = _safe_z(x)
    xbar = 1.0 - x
    lx = np.log(x)
    lxb = np.log(xbar)
    n1 = (
        x**3 * (-10.139 * lx**4 + 53.293 * lx**3 - 362.551 * lx**2 + 1069.36 * lx - 2080.42)
        + x**2 * (3.35283 * lx**4 + 38.0541 * lx**3 + 204.028 * lx**2 + 893.7 * lx + 1807.17)
        + x * (3.16049 * lx**4 + 27.0398 * lx**3 + 95.569 * lx**2 + 243.104 * lx + 167.326)
        - 0.197531 * lx**2
        - 8.49383 * lx
        + xbar**3 * (0.0903308 * lxb**3 - 0.534541 * lxb**2 + 1.48298 * lxb)
        + xbar**2 * (-0.344655 * lxb**3 - 1.88691 * lxb**2 - 4.7445 * lxb)
        + 2.107 * lxb**3
        + 10.3704 * lxb**2
        - 10.4458 * lxb
        + xbar * (-1.0535 * lxb**3 - 11.358 * lxb**2 + 3.08951 * lxb + 132.0)
        - 0.0981767 * x**6
        - 0.236655 * x**5
        + 65.9262 * x**4
        - 317.852
    )
    n2 = (
        x**3 * (0.431032 * lx**3 - 3.50262 * lx**2 + 9.32163 * lx - 8.91467)
        + x**2 * (-0.62563 * lx**3 - 2.59739 * lx**2 - 1.44432 * lx + 15.2779)
        + x * (-0.658436 * lx**3 - 3.29218 * lx**2 - 7.2428 * lx - 6.71585)
        + 0.395062 * lx
        + 0.00672043 * x**6
        - 0.0760913 * x**5
        + 0.55363 * x**4
        - 6.05761 * xbar
        + 15.8093
    )
    n0 = (
        x**3 * (32.8538 * lx**5 - 59.8157 * lx**4 + 1263.35 * lx**3 - 1593.34 * lx**2 + 9101.99 * lx + 965.588)
        + x**2 * (-3.03843 * lx**5 - 67.7124 * lx**4 - 364.242 * lx**3 - 1888.44 * lx**2 - 2473.02 * lx + 836.39)
        + x * (-2.64691 * lx**5 - 37.1111 * lx**4 - 137.094 * lx**3 - 453.207 * lx**2 - 173.141 * lx + 1583.11)
        + 3.25926 * lx**2
        + 32.5926 * lx
        + xbar**3 * (20.7632 * lxb**3 - 16.7047 * lxb**2 + 301.823 * lxb)
        + xbar**2 * (6.41987 * lxb**3 + 1.21958 * lxb**2 + 202.422 * lxb)
        - 34.7654 * lxb**3
        - 5.09037 * lxb**2
        + 637.843 * lxb
        + xbar * (-7.90123 * lxb**3 + 110.483 * lxb**2 - 62.9396 * lxb - 863.968)
        - 8.32785 * x**6
        + 84.9876 * x**5
        - 3254.01 * x**4
        + 948.568
    )
    return n0 + Nf * n1 + Nf**2 * n2


def _i3_scale_independent_matrix(x_ls_mat, y_ls_mat, Nf=3):
    plus = _build_plus_kernel_matrix(x_ls_mat, y_ls_mat, lambda z: _i3_qq_plus_fit_coeff(Nf=Nf) / (1.0 - z))
    regular = _build_regular_kernel_matrix(x_ls_mat, y_ls_mat, lambda z: _i3_qq_regular_piece_fit(z, Nf=Nf))
    return plus + regular


def _gamma_cusp_q(order, Nf=3):
    if order == 0:
        return 4.0 * CF
    if order == 1:
        return 4.0 * CF * ((67.0 / 9.0 - np.pi**2 / 3.0) * CA - 20.0 / 9.0 * TF * Nf)
    if order == 2:
        return 4.0 * CF * (
            CA**2 * (245.0 / 6.0 - 134.0 * np.pi**2 / 27.0 + 11.0 * np.pi**4 / 45.0 + 22.0 * zeta(3) / 3.0)
            + CA * TF * Nf * (-418.0 / 27.0 + 40.0 * np.pi**2 / 27.0 - 56.0 * zeta(3) / 3.0)
            + CF * TF * Nf * (-55.0 / 3.0 + 16.0 * zeta(3))
            - 16.0 * TF**2 * Nf**2 / 27.0
        )
    raise NotImplementedError("Quark cusp is implemented through three loops.")


def _gamma_beam_q(order, Nf=3):
    if order == 0:
        return 3.0 * CF
    if order == 1:
        return CF * (
            CF * (3.0 / 2.0 - 12.0 * zeta(2) + 24.0 * zeta(3))
            + CA * (17.0 / 6.0 + 44.0 * zeta(2) / 3.0 - 12.0 * zeta(3))
            + TF * Nf * (-2.0 / 3.0 - 16.0 * zeta(2) / 3.0)
        )
    if order == 2:
        return (
            CA * CF * TF * Nf * (-2672.0 * zeta(2) / 27.0 + 400.0 * zeta(3) / 9.0 + 4.0 * zeta(4) + 40.0)
            + CF**2 * TF * Nf * (40.0 * zeta(2) / 3.0 - 272.0 * zeta(3) / 3.0 + 232.0 * zeta(4) / 3.0 - 46.0)
            + CF * TF**2 * Nf**2 * (320.0 * zeta(2) / 27.0 - 64.0 * zeta(3) / 9.0 - 68.0 / 9.0)
            + CA**2 * CF * (16.0 * zeta(2) * zeta(3) - 410.0 * zeta(2) / 3.0 + 844.0 * zeta(3) / 3.0 - 494.0 * zeta(4) / 3.0 + 120.0 * zeta(5) + 151.0 / 4.0)
            + CA * CF**2 * (4496.0 * zeta(2) / 27.0 - 1552.0 * zeta(3) / 9.0 - 5.0 * zeta(4) + 40.0 * zeta(5) - 1657.0 / 36.0)
            + CF**3 * (-32.0 * zeta(2) * zeta(3) + 18.0 * zeta(2) + 68.0 * zeta(3) + 144.0 * zeta(4) - 240.0 * zeta(5) + 29.0 / 2.0)
        )
    raise NotImplementedError("Quark beam anomalous dimension is implemented through three loops.")


def _gamma_rapidity_q(order, Nf=3):
    if order == 0:
        return 0.0
    if order == 1:
        return CF * (CA * (-404.0 / 27.0 + 14.0 * zeta(3)) + TF * Nf * 112.0 / 27.0)
    if order == 2:
        return CF * (
            CA * TF * Nf * (-824.0 * zeta(2) / 81.0 - 904.0 * zeta(3) / 27.0 + 20.0 * zeta(4) / 3.0 + 62626.0 / 729.0)
            + CA**2 * (-88.0 * zeta(2) * zeta(3) / 3.0 + 3196.0 * zeta(2) / 81.0 + 6164.0 * zeta(3) / 27.0 + 77.0 * zeta(4) / 3.0 - 96.0 * zeta(5) - 297029.0 / 1458.0)
            + CF * TF * Nf * (-304.0 * zeta(3) / 9.0 - 16.0 * zeta(4) + 1711.0 / 27.0)
            + TF**2 * Nf**2 * (-64.0 * zeta(3) / 9.0 - 3712.0 / 729.0)
        )
    raise NotImplementedError("Quark rapidity anomalous dimension is implemented through three loops.")


_TRANS_MATRIX_CACHE = {}


def _trans_matrix_cache_key(x_ls_mat, y_ls_mat, Nf):
    return (
        tuple(np.asarray(x_ls_mat, dtype=float).round(15)),
        tuple(np.asarray(y_ls_mat, dtype=float).round(15)),
        int(Nf),
        str(_reference_root().resolve()),
    )


def _trans_fixed_matrices(x_ls_mat, y_ls_mat, Nf=3):
    key = _trans_matrix_cache_key(x_ls_mat, y_ls_mat, Nf)
    if key not in _TRANS_MATRIX_CACHE:
        identity = _identity_kernel(x_ls_mat, y_ls_mat)
        P0 = _splitting_kernel_matrix(x_ls_mat, y_ls_mat, 1, Nf)
        P1 = _splitting_kernel_matrix(x_ls_mat, y_ls_mat, 2, Nf)
        P2 = _splitting_kernel_matrix(x_ls_mat, y_ls_mat, 3, Nf)
        I1_hat = np.zeros_like(identity)
        I2_hat = _i2_scale_independent_matrix(x_ls_mat, y_ls_mat, Nf=Nf)
        I3_hat = _i3_scale_independent_matrix(x_ls_mat, y_ls_mat, Nf=Nf)
        _TRANS_MATRIX_CACHE[key] = (identity, P0, P1, P2, I1_hat, I2_hat, I3_hat)
    return _TRANS_MATRIX_CACHE[key]


def _trans_constants(Nf: int, LQ: float) -> dict[str, float]:
    gamma_cusp0 = _gamma_cusp_q(0, Nf)
    gamma_cusp1 = _gamma_cusp_q(1, Nf)
    gamma_cusp2 = _gamma_cusp_q(2, Nf)
    gamma_beam0 = _gamma_beam_q(0, Nf)
    gamma_beam1 = _gamma_beam_q(1, Nf)
    gamma_beam2 = _gamma_beam_q(2, Nf)
    return {
        "beta0": beta(0, Nf),
        "beta1": beta(1, Nf),
        "gamma_cusp0": gamma_cusp0,
        "gamma_cusp1": gamma_cusp1,
        "gamma_cusp2": gamma_cusp2,
        "gamma_beam0": gamma_beam0,
        "gamma_beam1": gamma_beam1,
        "gamma_beam2": gamma_beam2,
        "gamma_rap0": _gamma_rapidity_q(0, Nf),
        "gamma_rap1": _gamma_rapidity_q(1, Nf),
        "gamma_rap2": _gamma_rapidity_q(2, Nf),
        "A0": 2.0 * gamma_beam0 - gamma_cusp0 * LQ,
        "A1": 2.0 * gamma_beam1 - gamma_cusp1 * LQ,
        "A2": 2.0 * gamma_beam2 - gamma_cusp2 * LQ,
    }


def _trans_nlo_rows(identity_rows, P0_rows, I1_hat_rows, Lperp: float, LQ: float, c: dict[str, float]):
    one_loop_delta_rows = (
        -CF * Lperp**2
        - c["gamma_cusp0"] * Lperp * LQ / 2.0
        - CF * np.pi**2 / 6.0
    ) * identity_rows
    return (
        (c["gamma_beam0"] * Lperp + c["gamma_rap0"] * LQ) * identity_rows
        - Lperp * P0_rows
        + I1_hat_rows
        + one_loop_delta_rows
    )


def _trans_nnlo_rows(identity_rows, P0, P0_rows, P1_rows, I1_hat_rows, I2_hat_rows, Lperp: float, LQ: float, c: dict[str, float]):
    I2_delta_rows = (
        c["A0"] * (c["A0"] + 2.0 * c["beta0"]) * Lperp**2 / 8.0
        + ((c["A0"] + 2.0 * c["beta0"]) * c["gamma_rap0"] * LQ / 2.0 - c["gamma_cusp1"] * LQ / 2.0 + c["gamma_beam1"]) * Lperp
        + c["gamma_rap0"]**2 * LQ**2 / 2.0
        + c["gamma_rap1"] * LQ
    ) * identity_rows
    I2_lperp2_rows = (
        P0_rows @ P0 / 2.0
        + P0_rows * (c["gamma_cusp0"] * LQ - 2.0 * c["gamma_beam0"] - c["beta0"]) / 2.0
    ) * Lperp**2
    I2_lperp_rows = (
        -P1_rows
        - P0_rows * c["gamma_rap0"] * LQ
        - I1_hat_rows @ P0
        + (-c["gamma_cusp0"] * LQ / 2.0 + c["gamma_beam0"] + c["beta0"]) * I1_hat_rows
    ) * Lperp
    return I2_delta_rows + I2_lperp2_rows + I2_lperp_rows + c["gamma_rap0"] * LQ * I1_hat_rows + I2_hat_rows


def _trans_n3lo_rows(identity_rows, P0, P1, P0_rows, P1_rows, P2_rows, I1_hat_rows, I2_hat_rows, I3_hat_rows, Lperp: float, LQ: float, c: dict[str, float]):
    P0P0_rows = P0_rows @ P0
    I3_lperp3_rows = (
        (c["beta0"] / 2.0 + c["A0"] / 4.0) * P0P0_rows
        - (P0P0_rows @ P0) / 6.0
        + (c["beta0"]**2 * c["A0"] / 6.0 + c["beta0"] * c["A0"]**2 / 8.0 + c["A0"]**3 / 48.0) * identity_rows
        + (-c["beta0"] * c["A0"] / 2.0 - c["beta0"]**2 / 3.0 - c["A0"]**2 / 8.0) * P0_rows
    ) * Lperp**3
    I3_lperp2_rows = (
        (P0_rows @ P1) / 2.0
        + (P1_rows @ P0) / 2.0
        + (-c["beta1"] / 2.0 - c["A1"] / 2.0) * P0_rows
        + (c["beta1"] * c["A0"] / 4.0 + c["beta0"] * c["A1"] / 2.0 + c["A0"] * c["A1"] / 4.0) * identity_rows
        + (-c["beta0"] - c["A0"] / 2.0) * P1_rows
    ) * Lperp**2
    I3_lperp_rows = (
        -I2_hat_rows @ P0
        - P2_rows
        - P0_rows * c["gamma_rap1"] * LQ
        + (2.0 * c["beta0"] * c["gamma_rap1"] * LQ + c["gamma_rap1"] * c["A0"] * LQ / 2.0 + c["A2"] / 2.0) * identity_rows
        + (2.0 * c["beta0"] + c["A0"] / 2.0) * I2_hat_rows
    ) * Lperp
    return I3_lperp3_rows + I3_lperp2_rows + I3_lperp_rows + c["gamma_rap2"] * LQ * identity_rows + c["gamma_rap1"] * LQ * I1_hat_rows + I3_hat_rows


def transversity_matching_kernel_rows(x_ls_mat, y_ls_mat, row_indices, b_fm, mu=2.0, zeta=4.0, Nf=3, order="N3LO", isov=True):
    """Selected rows of the transversity isoV q<-q TMDPDF matching kernel."""
    if not isov:
        raise NotImplementedError("Only the positive-x q<-q transversity isoV kernel is implemented.")
    _check_identical_grids(x_ls_mat, y_ls_mat)

    row_indices = list(row_indices)
    max_power = _max_power(order)
    identity, P0, P1, P2, I1_hat, I2_hat, I3_hat = _trans_fixed_matrices(x_ls_mat, y_ls_mat, Nf=Nf)
    identity_rows = identity[row_indices, :]
    P0_rows = P0[row_indices, :]
    P1_rows = P1[row_indices, :]
    P2_rows = P2[row_indices, :]
    I1_hat_rows = I1_hat[row_indices, :]
    I2_hat_rows = I2_hat[row_indices, :]
    I3_hat_rows = I3_hat[row_indices, :]

    a_s = alphas_nloop(mu, order=2, Nf=Nf) / (4.0 * np.pi)
    Lperp = Lz_func(b_fm, mu)
    LQ = np.log(zeta / mu**2)
    constants = _trans_constants(Nf, LQ)

    I1_rows = _trans_nlo_rows(identity_rows, P0_rows, I1_hat_rows, Lperp, LQ, constants)
    rows = identity_rows + a_s * I1_rows
    if max_power == 1:
        return rows

    I2_rows = _trans_nnlo_rows(
        identity_rows,
        P0,
        P0_rows,
        P1_rows,
        I1_hat_rows,
        I2_hat_rows,
        Lperp,
        LQ,
        constants,
    )
    rows = rows + a_s**2 * I2_rows
    if max_power == 2:
        return rows

    I3_rows = _trans_n3lo_rows(
        identity_rows,
        P0,
        P1,
        P0_rows,
        P1_rows,
        P2_rows,
        I1_hat_rows,
        I2_hat_rows,
        I3_hat_rows,
        Lperp,
        LQ,
        constants,
    )
    return rows + a_s**3 * I3_rows
