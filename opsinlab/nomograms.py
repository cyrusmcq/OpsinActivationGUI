"""
opsinlab.nomograms
------------------
Govardovskii et al. (2000) photopigment templates and helpers.

Pure functions only (no plotting, no I/O). Designed to pair with opsinlab.io_led.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable

# --- Govardovskii 2000 constants (as used in prior scripts) ---
A = 69.7
B = 28.0
C = -14.9
D = 0.674
b = 0.922
c = 1.104

# --- Species → opsin λmax table (nm) ---
# Keys standardized to {Rh, LW, MW, SW, UVS}; only relevant ones present per species.
species_data: Dict[str, Dict[str, float]] = {
    "Guinea Pig": {
        "Rh": 500,
        "MW": 529,
        "SW": 430,
    },
    "Macaque": {
        "Rh": 503,
        "LW": 561,
        "MW": 536,
        "SW": 430,
    },
    "Mouse": {
        "Rh": 498,
        "MW": 508,
        "UVS": 360,   # mouse short-wave is UV-sensitive
    },
}

# ---------------- Template functions ----------------

def corrected_a(lmax: float) -> float:
    """Govardovskii corrected 'a' parameter as a function of λmax (nm)."""
    return 0.8795 + 0.0459 * np.exp(-((lmax - 300.0) ** 2) / 11940.0)

def alpha_band(l: np.ndarray, lmax: float) -> np.ndarray:
    """Main (α) band of the template."""
    l = np.asarray(l, dtype=float)
    lr = lmax / l
    a = corrected_a(lmax)
    return 1.0 / (np.exp(A * (a - lr)) + np.exp(B * (b - lr)) + np.exp(C * (c - lr)) + D)

def beta_band(l: np.ndarray, lmax: float) -> np.ndarray:
    """Secondary (β) band of the template."""
    l = np.asarray(l, dtype=float)
    return 0.26 * np.exp(-((l - (189.0 + 0.315 * lmax)) / (-40.5 + 0.195 * lmax)) ** 2)

def nomogram(l: np.ndarray, lmax: float) -> np.ndarray:
    """Full Govardovskii nomogram (α + β)."""
    return alpha_band(l, lmax) + beta_band(l, lmax)

def normalize_curve(curve: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Area-normalize a curve (∫ curve dλ = 1 over provided wavelengths)."""
    curve = np.asarray(curve, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    # Use only finite values for the integral
    mask = np.isfinite(curve) & np.isfinite(wavelengths)
    if mask.sum() > 1:
        area = np.trapezoid(curve[mask], wavelengths[mask])
        if area > 0:
            return curve / area
    return curve

# ---------------- Public helpers ----------------

def available_species() -> list[str]:
    """List species names shipped in species_data."""
    return list(species_data.keys())

def species_opsins(species_name: str) -> list[str]:
    """List opsin keys available for a species."""
    if species_name not in species_data:
        raise KeyError(f"Unknown species '{species_name}'. Use available_species().")
    return list(species_data[species_name].keys())

def generate_nomograms(
    species_name: str,
    wavelengths: Iterable[float],
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Generate (optionally area-normalized) nomograms for all opsins in a species.

    Parameters
    ----------
    species_name : str
        One of available_species().
    wavelengths : Iterable[float]
        Wavelength grid (nm), typically 300–700 from your LED pipeline.
    normalize : bool
        If True, each opsin curve is area-normalized (∫ = 1).

    Returns
    -------
    pd.DataFrame
        Columns: ['Wavelength', <opsin1>, <opsin2>, ...]
        Only newly computed columns are returned (clean frame).
    """
    if species_name not in species_data:
        raise KeyError(f"Unknown species '{species_name}'. Use available_species().")

    wl = np.asarray(wavelengths, dtype=float)
    out = {"Wavelength": wl}

    for opsin, lmax in species_data[species_name].items():
        curve = nomogram(wl, lmax)
        out[opsin] = normalize_curve(curve, wl) if normalize else curve

    return pd.DataFrame(out)

def interpolate_nomograms(
    nomo_df: pd.DataFrame,
    target_wavelengths: Iterable[float],
) -> pd.DataFrame:
    """
    Interpolate an existing nomogram DataFrame onto a new wavelength grid.

    Returns a clean DataFrame with ['Wavelength', opsin...].
    """
    target = np.asarray(target_wavelengths, dtype=float)
    out = {"Wavelength": target}
    for col in nomo_df.columns:
        if col == "Wavelength":
            continue
        out[col] = np.interp(target, nomo_df["Wavelength"].values, nomo_df[col].values)
    return pd.DataFrame(out)
