"""
Activation & isolation math for OpsinActivation.

This module builds:
- ActivationMatrix: LED × Opsin (integral of photon_flux(λ) * opsin(λ))
- InverseActivationMatrix: (A_3x3^T)^-1 or pinv over selected 3 LEDs × 3 opsins
- ProductMatrix: raw LED vectors that isolate each opsin (columns)
- ScaledProductMatrix: scaled version per chosen strategy
"""

from __future__ import annotations
from typing import List, Dict, Literal
import numpy as np
import pandas as pd


# ----------------------------- Core helpers -----------------------------

def _validate_columns(df: pd.DataFrame, need: List[str], df_name: str) -> None:
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing columns: {missing}")


def compute_led_opsin_activations(
    photon_flux_df: pd.DataFrame,
    nomograms_df: pd.DataFrame,
    leds: List[str],
    opsins: List[str],
    wl_min: float = 300.0,
    wl_max: float = 700.0,
) -> pd.DataFrame:
    """
    Multiply LED photon flux spectra by opsin nomograms and integrate over wavelength.
    Returns a long dataframe with columns: ['LED', 'Opsin', 'Activation'].
    """
    _validate_columns(photon_flux_df, ["Wavelength"], "photon_flux_df")
    _validate_columns(nomograms_df, ["Wavelength"], "nomograms_df")

    # Trim wavelength range and align on wavelength
    pf = photon_flux_df[(photon_flux_df["Wavelength"] >= wl_min) &
                        (photon_flux_df["Wavelength"] <= wl_max)].copy()
    nm = nomograms_df[(nomograms_df["Wavelength"] >= wl_min) &
                      (nomograms_df["Wavelength"] <= wl_max)].copy()

    # Inner-join on wavelength to ensure identical support
    merged = pd.merge(pf, nm, on="Wavelength", how="inner", suffixes=("_pf", "_nm"))
    if merged.empty:
        raise ValueError("No overlapping wavelengths between photon flux and nomograms in the specified range.")

    # Ensure requested LEDs/opsins exist
    missing_leds = [l for l in leds if l not in pf.columns]
    missing_ops = [o for o in opsins if o not in nm.columns]
    if missing_leds:
        raise KeyError(f"Requested LED(s) not found in photon_flux_df: {missing_leds}")
    if missing_ops:
        raise KeyError(f"Requested opsin(s) not found in nomograms_df: {missing_ops}")

    wl = merged["Wavelength"].to_numpy()
    out_rows = []
    for led in leds:
        led_spec = merged[led].to_numpy()
        for ops in opsins:
            ops_spec = merged[ops].to_numpy()
            # Photon flux * sensitivity (dimension: photons·nm^-1 * unitless)
            prod = led_spec * ops_spec
            # Integrate over λ (trapezoid, wavelength in nm)
            activation = np.trapezoid(prod, wl)
            out_rows.append((led, ops, float(activation)))

    return pd.DataFrame(out_rows, columns=["LED", "Opsin", "Activation"])


def to_activation_matrix(activations_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the long table (LED, Opsin, Activation) into LED×Opsin.
    """
    _validate_columns(activations_long, ["LED", "Opsin", "Activation"], "activations_long")
    A = activations_long.pivot(index="LED", columns="Opsin", values="Activation").fillna(0.0)
    # Stable ordering by name helps reproducibility
    A = A.sort_index(axis=0).sort_index(axis=1)
    return A


def build_activation_matrix(
    photon_flux_df: pd.DataFrame,
    nomograms_df: pd.DataFrame,
    leds: List[str],
    opsins: List[str],
    wl_min: float = 300.0,
    wl_max: float = 700.0,
) -> pd.DataFrame:
    """
    Convolve LED photon flux spectra with opsin nomograms and integrate to get:
    ActivationMatrix  (LEDs as rows, Opsins as columns)
    """
    # Rename *_PhotonFlux to canonical LED names
    photon_flux_df = photon_flux_df.rename(
        columns={
            "Red_PhotonFlux": "Red",
            "Green_PhotonFlux": "Green",
            "Blue_PhotonFlux": "Blue",
            "UV_PhotonFlux": "UV",
        }
    )

    res_long = compute_led_opsin_activations(
        photon_flux_df, nomograms_df, leds=leds, opsins=opsins, wl_min=wl_min, wl_max=wl_max
    )
    A = to_activation_matrix(res_long)
    A = A.reindex(index=leds, columns=opsins)
    return A



def inverse_activation_matrix(
    A_full: pd.DataFrame,
    leds3: List[str],
    opsins3: List[str],
    method: Literal["inv", "pinv"] = "inv",
) -> np.ndarray:
    """
    Pick a 3×3 block (LEDs3 × Opsins3), transpose to M = A^T (Opsins×LEDs),
    and return inverse or pseudoinverse.
    """
    for l in leds3:
        if l not in A_full.index:
            raise KeyError(f"LED '{l}' not in ActivationMatrix rows.")
    for o in opsins3:
        if o not in A_full.columns:
            raise KeyError(f"Opsin '{o}' not in ActivationMatrix columns.")

    A3 = A_full.loc[leds3, opsins3].to_numpy()   # (3,3)
    if A3.shape != (3, 3):
        raise ValueError(f"Expected a 3×3 submatrix, got {A3.shape}.")
    M = A3.T                                      # (3,3)

    if method == "inv":
        return np.linalg.inv(M)
    return np.linalg.pinv(M)


def product_matrix_from_inverse(
    inverse_M: np.ndarray,
    opsins3: List[str],
    leds3: List[str],
) -> pd.DataFrame:
    """
    inverse_M is (LED × Opsin). Each column is the LED vector that isolates that opsin.
    Return ProductMatrix with rows=LEDs, columns='<opsin>_ISO'.
    """
    if inverse_M.shape != (3, 3):
        raise ValueError(f"inverse_M must be 3×3, got {inverse_M.shape}")

    # Columns = opsins, rows = LEDs
    cols = [f"{op}_ISO" for op in opsins3]
    df = pd.DataFrame(inverse_M, index=leds3, columns=opsins3)
    df.columns = cols
    return df


    # inverse_M maps Opsin->LED weights directly (columns are iso targets).
    X_opsin_iso = inverse_M @ np.eye(3)  # (3×3)
    df = pd.DataFrame(X_opsin_iso, index=opsins3, columns=leds3).T
    df.columns = [f"{op}_ISO" for op in opsins3]
    return df


def scale_product_matrix(
    product_df: pd.DataFrame,
    mode: Literal["match_across_isolations", "per_isolation_max"] = "match_across_isolations",
) -> pd.DataFrame:
    """
    Two scaling options:
      - 'match_across_isolations': divide the entire 3×3 by the global max(|value|)
      - 'per_isolation_max': divide each column by its own max(|value|)
    """
    M = product_df.copy()
    if mode == "match_across_isolations":
        s = float(np.max(np.abs(M.to_numpy()))) if M.size else 1.0
        if s == 0.0:
            s = 1.0
        return M / s
    # per-column
    for c in M.columns:
        s = float(np.max(np.abs(M[c].to_numpy()))) if len(M[c]) else 1.0
        if s == 0.0:
            s = 1.0
        M[c] = M[c] / s
    return M


# ----------------------------- Pipeline -----------------------------

_SPECIES_OPS3: Dict[str, List[str]] = {
    "macaque": ["LW", "MW", "SW"],
    "mouse":   ["Rh", "MW", "UVS"],  # or ["Rh","MW","SW"] if your nomogram uses SW
    "guinea pig": ["Rh", "MW", "SW"],
    "guinea_pig": ["Rh", "MW", "SW"],
    "guineapig":  ["Rh", "MW", "SW"],
}


def isolation_pipeline(
    photon_flux_df: pd.DataFrame,
    nomograms_df: pd.DataFrame,
    species: str,
    selected_leds_3: List[str],
    strategy: Literal["match_across_isolations", "per_isolation_max"] = "match_across_isolations",
    use_exact_inverse: bool = True,
    wl_min: float = 300.0,
    wl_max: float = 700.0,
) -> Dict[str, object]:
    """
    Returns a dict with:
      - ActivationMatrix (LED×Opsin, all opsins present in nomograms_df)
      - InverseActivationMatrix (3×3)
      - ProductMatrix (3×3, raw)
      - ScaledProductMatrix (3×3, scaled)
      - Modulations (3×3, same as ScaledProductMatrix but nicer column names)
    """
    sp = species.strip().lower()
    if sp not in _SPECIES_OPS3:
        # fallback: macaque scheme
        ops3 = ["LW", "MW", "SW"]
    else:
        ops3 = _SPECIES_OPS3[sp]

    # All opsins available in nomograms_df (except Wavelength)
    all_opsins = [c for c in nomograms_df.columns if c != "Wavelength"]
    if not all_opsins:
        raise ValueError("No opsin columns found in nomograms_df.")

    # Build full ActivationMatrix over the selected LEDs and all opsins
    A = build_activation_matrix(
        photon_flux_df, nomograms_df,
        leds=selected_leds_3, opsins=all_opsins,
        wl_min=wl_min, wl_max=wl_max
    )

    # Compute 3×3 inverse over the chosen opsins
    invM = inverse_activation_matrix(A, selected_leds_3, ops3, method=("inv" if use_exact_inverse else "pinv"))

    # Product matrix (raw LED vectors per isolation)
    P = product_matrix_from_inverse(invM, ops3, selected_leds_3)

    # Apply scaling strategy
    P_scaled = scale_product_matrix(P, mode=strategy)

    nice = P_scaled.copy()
    nice.columns = [c.replace("_ISO", "_isolating") for c in nice.columns]

    return {
        "ActivationMatrix": A,               # LED×Opsin (all)
        "InverseActivationMatrix": invM,     # 3×3 (Opsins×LEDs)
        "ProductMatrix": P,                  # 3×3 LED×<opsin>_ISO (raw)
        "ScaledProductMatrix": P_scaled,     # 3×3 LED×<opsin>_ISO (scaled)
        "Modulations": nice,                 # LEDs × '<opsin>_isolating'
        "opsins3": ops3,
        "leds3": selected_leds_3,
        "strategy": strategy,
        "use_exact_inverse": use_exact_inverse,
    }


# ----------------------------- Back-compat alias -----------------------------

def activation_pipeline(*args, **kwargs):
    """Backward-compat: old name -> new isolation_pipeline."""
    return isolation_pipeline(*args, **kwargs)


__all__ = [
    "compute_led_opsin_activations",
    "to_activation_matrix",
    "build_activation_matrix",
    "inverse_activation_matrix",
    "product_matrix_from_inverse",
    "scale_product_matrix",
    "isolation_pipeline",
    "activation_pipeline",
]
