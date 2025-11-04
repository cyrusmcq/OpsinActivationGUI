"""
Activation & isolation math for OpsinActivation.

Builds:
- ActivationMatrix: LED × Opsin  (∫ φ(λ) · S(λ) dλ) in photons / (µm²·s) as seen by each opsin
- InverseActivationMatrix: (A_3x3^T)^-1 or pinv over selected 3 LEDs × 3 opsins
- ProductMatrix: raw LED vectors that isolate each opsin (columns)
- ScaledProductMatrix: scaled version per chosen strategy
- Opsin-weighted R* estimates: convert opsin-weighted photons/(µm²·s) to R*/s using species collecting areas
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Literal, Optional
import numpy as np
import pandas as pd

# ----------------------------- Core helpers -----------------------------
def _validate_columns(df: pd.DataFrame, need: List[str], df_name: str) -> None:
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing columns: {missing}")

def _canonicalize_flux_columns(photon_flux_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy where LED flux columns exist under canonical names: 'Red','Green','Blue','UV'.
    Accepts either '*_PhotonFlux' or bare LED names. Non-present LEDs are ignored.
    """
    df = photon_flux_df.copy()
    mapping = {
        "Red_PhotonFlux": "Red",
        "Green_PhotonFlux": "Green",
        "Blue_PhotonFlux": "Blue",
        "UV_PhotonFlux": "UV",
    }
    for src, dst in mapping.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    return df

# ----------------------------- Activation math -----------------------------
def compute_led_opsin_activations(
    photon_flux_df: pd.DataFrame,
    nomograms_df: pd.DataFrame,
    leds: List[str],
    opsins: List[str],
    wl_min: float = 300.0,
    wl_max: float = 700.0,
) -> pd.DataFrame:
    """
    Multiply LED photon-flux spectra by opsin nomograms and integrate over wavelength.
    Returns a long DataFrame with columns: ['LED', 'Opsin', 'Activation'].
    Units: photons/(µm²·s) per LED-opsin pair.
    """
    _validate_columns(photon_flux_df, ["Wavelength"], "photon_flux_df")
    _validate_columns(nomograms_df, ["Wavelength"], "nomograms_df")

    pf0 = _canonicalize_flux_columns(photon_flux_df)

    pf = pf0[(pf0["Wavelength"] >= wl_min) & (pf0["Wavelength"] <= wl_max)].copy()
    nm = nomograms_df[(nomograms_df["Wavelength"] >= wl_min) & (nomograms_df["Wavelength"] <= wl_max)].copy()

    merged = pd.merge(pf, nm, on="Wavelength", how="inner", suffixes=("_pf", "_nm"))
    if merged.empty:
        raise ValueError("No overlapping wavelengths between photon flux and nomograms in the specified range.")

    missing_leds = [l for l in leds if l not in pf.columns]
    missing_ops  = [o for o in opsins if o not in nm.columns]
    if missing_leds:
        raise KeyError(f"Requested LED(s) not found in photon_flux_df: {missing_leds}")
    if missing_ops:
        raise KeyError(f"Requested opsin(s) not found in nomograms_df: {missing_ops}")

    wl = merged["Wavelength"].to_numpy(float)
    out_rows = []
    for led in leds:
        led_spec = merged[led].to_numpy(float)
        for ops in opsins:
            ops_spec = merged[ops].to_numpy(float)
            prod = led_spec * ops_spec
            activation = float(np.trapezoid(prod, wl))
            out_rows.append((led, ops, activation))

    return pd.DataFrame(out_rows, columns=["LED", "Opsin", "Activation"])

def to_activation_matrix(activations_long: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(activations_long, ["LED", "Opsin", "Activation"], "activations_long")
    A = activations_long.pivot(index="LED", columns="Opsin", values="Activation").fillna(0.0)
    return A.sort_index(axis=0).sort_index(axis=1)

def build_activation_matrix(
    photon_flux_df: pd.DataFrame,
    nomograms_df: pd.DataFrame,
    leds: List[str],
    opsins: List[str],
    wl_min: float = 300.0,
    wl_max: float = 700.0,
) -> pd.DataFrame:
    res_long = compute_led_opsin_activations(
        photon_flux_df, nomograms_df, leds=leds, opsins=opsins, wl_min=wl_min, wl_max=wl_max
    )
    A = to_activation_matrix(res_long)
    return A.reindex(index=leds, columns=opsins)

def inverse_activation_matrix(
    A_full: pd.DataFrame,
    leds3: List[str],
    opsins3: List[str],
    method: Literal["inv", "pinv"] = "inv",
) -> np.ndarray:
    for l in leds3:
        if l not in A_full.index:
            raise KeyError(f"LED '{l}' not in ActivationMatrix rows.")
    for o in opsins3:
        if o not in A_full.columns:
            raise KeyError(f"Opsin '{o}' not in ActivationMatrix columns.")

    A3 = A_full.loc[leds3, opsins3].to_numpy(float)   # (3,3)
    if A3.shape != (3, 3):
        raise ValueError(f"Expected a 3×3 submatrix, got {A3.shape}.")
    M = A3.T  # (3,3)

    if method == "inv":
        return np.linalg.inv(M)
    return np.linalg.pinv(M)

def product_matrix_from_inverse(
    inverse_M: np.ndarray,
    opsins3: List[str],
    leds3: List[str],
) -> pd.DataFrame:
    if inverse_M.shape != (3, 3):
        raise ValueError(f"inverse_M must be 3×3, got {inverse_M.shape}")
    df = pd.DataFrame(inverse_M, index=leds3, columns=opsins3)
    df.columns = [f"{op}_ISO" for op in opsins3]
    return df

def scale_product_matrix(
    product_df: pd.DataFrame,
    mode: Literal["match_across_isolations", "per_isolation_max"] = "match_across_isolations",
) -> pd.DataFrame:
    M = product_df.copy()
    if mode == "match_across_isolations":
        s = float(np.max(np.abs(M.to_numpy()))) if M.size else 1.0
        return M / (s if s != 0.0 else 1.0)
    for c in M.columns:
        s = float(np.max(np.abs(M[c].to_numpy()))) if len(M[c]) else 1.0
        M[c] = M[c] / (s if s != 0.0 else 1.0)
    return M

# ----------------------------- Species opsin sets -----------------------------
_SPECIES_OPS3: Dict[str, List[str]] = {
    "macaque": ["LW", "MW", "SW"],
    "mouse":   ["Rh", "MW", "UVS"],
    "guinea pig": ["Rh", "MW", "SW"],
    "guinea_pig": ["Rh", "MW", "SW"],
    "guineapig":  ["Rh", "MW", "SW"],
}

# ----------------------------- Collecting areas (µm²) -----------------------------
# Values are conservative midpoints commonly used in the literature.
# DOIs included inline.
@dataclass(frozen=True)
class CollectingAreas:
    rod: float
    cone_LW: Optional[float] = None
    cone_MW: Optional[float] = None
    cone_SW: Optional[float] = None
    cone_UVS: Optional[float] = None

COLLECTING_AREAS_BY_SPECIES: Dict[str, CollectingAreas] = {
    # Mouse rods ~0.5–0.7 µm²; cones a bit smaller.
    # Naarendorp et al., J Neurosci 2010 (10.1523/JNEUROSCI.2186-10.2010)
    # Umino et al., J Neurosci 2008 (10.1523/JNEUROSCI.3551-07.2008)
    "Mouse": CollectingAreas(rod=0.6, cone_MW=0.5, cone_UVS=0.5),

    # Guinea pig midpoints.
    # Field et al., Neuron 2002 (10.1016/S0896-6273(02)00822-X)
    # Fain et al., F1000Research 2018 (10.12688/f1000research.14021.1)
    "Guinea Pig": CollectingAreas(rod=0.7, cone_MW=0.5, cone_SW=0.5),

    # Macaque rods ~1.0 µm²; cones ~0.6 µm².
    # Baylor, Nunn & Schnapf, J Physiol 1984 (10.1113/jphysiol.1984.sp015518)
    # Schneeweis & Schnapf, J Physiol 1999 (PMCID: PMC6786037)
    "Macaque": CollectingAreas(rod=1.0, cone_LW=0.6, cone_MW=0.6, cone_SW=0.6),
}

# ----------------------------- Opsin-weighted isomerizations -----------------------------
def opsin_weighted_isomerizations(
    photon_flux_df: pd.DataFrame,
    nomograms_df: pd.DataFrame,
    species: str,
    selected_leds_3: List[str],
    quantum_efficiency: float = 0.67,
    wl_min: float = 300.0,
    wl_max: float = 700.0,
) -> Dict[str, object]:
    """
    Compute R*/s per opsin using the opsin-weighted activation:
      1) A = ActivationMatrix (LED × Opsin) from attenuated photon flux `photon_flux_df`
      2) per-opsin photons/(µm²·s) = sum over *selected_leds_3*
      3) R* = photons/(µm²·s) × A_eff(opsin) × QE
    """
    sp = species.strip().lower()
    ops3 = _SPECIES_OPS3.get(sp, ["LW", "MW", "SW"])
    opsins_all = [c for c in nomograms_df.columns if c != "Wavelength"]

    A = build_activation_matrix(
        photon_flux_df=photon_flux_df,
        nomograms_df=nomograms_df,
        leds=selected_leds_3,
        opsins=opsins_all,
        wl_min=wl_min,
        wl_max=wl_max,
    )  # photons/(µm²·s)

    areas = COLLECTING_AREAS_BY_SPECIES.get(species.strip(), COLLECTING_AREAS_BY_SPECIES["Macaque"])
    opsin_to_area = {
        "Rh":  areas.rod,
        "LW":  areas.cone_LW,
        "MW":  areas.cone_MW,
        "SW":  areas.cone_SW,
        "UVS": areas.cone_UVS,
    }

    photons_per_opsin = A.sum(axis=0).to_dict()

    R_per_opsin: Dict[str, float] = {}
    for opsin, photons in photons_per_opsin.items():
        area = opsin_to_area.get(opsin, None)
        if area is not None:
            R_per_opsin[opsin] = float(photons * area * quantum_efficiency)

    perLED: Dict[str, Dict[str, float]] = {}
    for opsin in A.columns:
        area = opsin_to_area.get(opsin, None)
        if area is None:
            continue
        perLED[opsin] = {led: float(A.loc[led, opsin] * area * quantum_efficiency) for led in A.index}

    return {
        "A": A,
        "opsins3": ops3,
        "photons_um2_s_per_opsin": {k: float(v) for k, v in photons_per_opsin.items()},
        "Rstar_per_opsin": R_per_opsin,
        "Rstar_per_opsin_perLED": perLED,
    }

# ----------------------------- Pipeline -----------------------------
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
    ops3 = _SPECIES_OPS3.get(sp, ["LW", "MW", "SW"])

    all_opsins = [c for c in nomograms_df.columns if c != "Wavelength"]
    if not all_opsins:
        raise ValueError("No opsin columns found in nomograms_df.")

    A = build_activation_matrix(
        photon_flux_df, nomograms_df,
        leds=selected_leds_3, opsins=all_opsins,
        wl_min=wl_min, wl_max=wl_max
    )

    invM = inverse_activation_matrix(A, selected_leds_3, ops3, method=("inv" if use_exact_inverse else "pinv"))
    P = product_matrix_from_inverse(invM, ops3, selected_leds_3)
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

# --- Contrast helper (Weber at gray 50%) ---
def isolation_contrast_from_A_and_mod(
    A_full: pd.DataFrame,   # LED × Opsin (for the 3 selected LEDs at least)
    leds3: list[str],       # the trio in use, order matters
    target_opsin: str,      # e.g., "LW_isolating" -> target_opsin = "LW"
    mod_vec: np.ndarray,    # shape (3,), the column from Modulations for that target
    gray_level: float = 0.5 # per-LED gray
) -> dict:
    """
    Returns dict with: alpha_max, A_gray, d_on, d_off, contrast_weber (fraction),
    and a convenience percent field.
    """
    # 3×1 LED vector at gray
    d0 = np.full(3, float(gray_level))
    # 3×1 modulation
    m = np.asarray(mod_vec, dtype=float).reshape(3)

    # Clip-safe amplitude
    with np.errstate(divide="ignore", invalid="ignore"):
        per_led_limits = np.where(m != 0, (gray_level / np.abs(m)), np.inf)
    alpha_max = float(np.min(per_led_limits))

    # Use alpha_max by default (largest unclipped symmetric contrast)
    alpha = alpha_max

    # Pull A3 (LED×Opsin) restricted to the trio and the full set of opsins
    A3 = A_full.loc[leds3, :]
    # Extract the plain opsin key (strip "_isolating" if present)
    ops_key = target_opsin.replace("_isolating", "")
    if ops_key not in A3.columns:
        raise KeyError(f"Opsin '{ops_key}' not found in ActivationMatrix columns.")

    a_col = A3[ops_key].to_numpy(float)  # shape (3,)

    A_gray = float(a_col @ d0)
    delta  = float(a_col @ (alpha * m))
    A_on   = A_gray + delta
    A_off  = A_gray - delta

    # Weber (== Michelson here)
    contrast = delta / A_gray if A_gray > 0 else np.nan

    # Return LED drive vectors too (in case you want to display them)
    d_on  = d0 + alpha * m
    d_off = d0 - alpha * m

    return {
        "alpha_max": alpha_max,
        "A_gray": A_gray,
        "A_on": A_on,
        "A_off": A_off,
        "contrast_weber": contrast,
        "contrast_percent": (100.0 * contrast) if np.isfinite(contrast) else np.nan,
        "d_on": d_on,
        "d_off": d_off,
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
    "opsin_weighted_isomerizations",
    "COLLECTING_AREAS_BY_SPECIES",
    "CollectingAreas",
]

