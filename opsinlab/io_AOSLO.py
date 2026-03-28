"""
opsinlab.io_AOSLO
-----------------
Functions for loading AOSLO spectra (3 primaries: Red/Green/Blue)
and converting to photon flux using power-meter measurements.

Unlike io_led.py (which applies power-meter responsivity corrections),
the AOSLO spectra are already measured via spectroradiometer and only
need to be scaled by the measured power for each channel.

Assets used:
  - assets/AOSLO_spectra.csv   (spectroradiometer, arbitrary units, 3 channels)
  - assets/AOSLO_power.csv     (power-meter readings per channel in nW)

Pipeline
--------
1. load_aoslo_spectra()        – load spectroradiometer data (arb. units, 3 channels)
2. normalize_spectra()         – normalize each channel to area = 1
3. convert_to_photon_flux()    – scale by power and convert to photons/(s·µm²·nm)
4. process_aoslo_data()        – top-level convenience function
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# ── Asset paths ──────────────────────────────────────────────────────
_DEFAULT_ASSETS = (Path(__file__).resolve().parent.parent / "assets").resolve()
_ENV_OVERRIDE = os.environ.get("OPSIN_ASSETS_DIR")
ASSETS_DIR = Path(_ENV_OVERRIDE).resolve() if _ENV_OVERRIDE else _DEFAULT_ASSETS

SPECTRA_PATH = ASSETS_DIR / "AOSLO_spectra.csv"
POWER_CSV_PATH = ASSETS_DIR / "AOSLO_power.csv"

# Mapping from CSV column names to canonical LED color names
_CHANNEL_MAP = {
    "Red (680)": "Red",
    "Green (543)": "Green",
    "Blue (488)": "Blue",
}


def set_assets_dir(path: str | os.PathLike) -> None:
    """Optionally set/override the assets directory at runtime."""
    global ASSETS_DIR, SPECTRA_PATH, POWER_CSV_PATH
    ASSETS_DIR = Path(path).resolve()
    SPECTRA_PATH = ASSETS_DIR / "AOSLO_spectra.csv"
    POWER_CSV_PATH = ASSETS_DIR / "AOSLO_power.csv"


# ─────────────────────────────────────────────────────────────────────
# 1. Load spectroradiometer spectra (3 channels)
# ─────────────────────────────────────────────────────────────────────
def load_aoslo_spectra(file_path: Path = SPECTRA_PATH) -> pd.DataFrame:
    data = pd.read_csv(file_path, skiprows=3)
    wavelengths = data.iloc[:, 0].astype(float).values

    out = {"Wavelength": wavelengths}

    for csv_name, canon_name in _CHANNEL_MAP.items():
        if csv_name in data.columns:
            out[canon_name] = data[csv_name].astype(float).values
        else:
            out[canon_name] = np.zeros_like(wavelengths)

    df = pd.DataFrame(out)
    df = df.dropna(subset=["Wavelength"])
    return df


# ─────────────────────────────────────────────────────────────────────
# 2. Normalize each channel spectrum (area = 1)
# ─────────────────────────────────────────────────────────────────────
def normalize_spectra(spectra_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each channel's spectrum so that ∫ S(λ) dλ = 1.

    Parameters
    ----------
    spectra_df : pd.DataFrame
        Columns: ['Wavelength', 'Red', 'Green', 'Blue']

    Returns
    -------
    pd.DataFrame
        Columns: ['Wavelength', 'Red', 'Green', 'Blue']
        Each channel normalized to unit area.
    """
    wl = spectra_df["Wavelength"].values
    out = {"Wavelength": wl}

    for ch in ["Red", "Green", "Blue"]:
        if ch not in spectra_df.columns:
            out[ch] = np.zeros_like(wl)
            continue

        vals = spectra_df[ch].values.copy()
        mask = ~np.isnan(vals)

        if mask.sum() > 1:
            area = np.trapezoid(vals[mask], wl[mask])
            out[ch] = (vals / area) if area > 0 else np.zeros_like(vals)
        else:
            out[ch] = np.zeros_like(vals)

    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────
# 3. Convert normalized spectra → photon flux
# ─────────────────────────────────────────────────────────────────────
def convert_to_photon_flux(
    normalized_df: pd.DataFrame,
    led_powers_watts: dict,
    measurement_area_um2: float = 495.0 * 247.0,
) -> pd.DataFrame:
    """
    Convert normalized spectra to photon flux using measured powers.

    The power meter measures total power over the full-field stimulus area.
    To get flux (photons/s/µm²), we divide by the measurement area —
    i.e. the area over which the power was actually collected.

    For each channel with nonzero power:
        photon_flux(λ) = [P_total / <E>] / measurement_area × S_norm(λ)

    where <E> = ∫ E(λ)·S_norm(λ) dλ is the spectrum-weighted mean
    photon energy.

    Channels with zero power are included as all-zero columns.

    Parameters
    ----------
    normalized_df : pd.DataFrame
        Columns: ['Wavelength', 'Red', 'Green', 'Blue'], area-normalized.
    led_powers_watts : dict
        {'Red': watts, 'Green': watts, 'Blue': watts}.
        Zero or missing entries produce zero-flux columns.
    measurement_area_um2 : float
        Area over which the power meter measured total power (µm²).
        Default: 495.0 * 247.0 = 122265 µm² (AOSLO full-field rectangle ON THE RETINA).

    Returns
    -------
    pd.DataFrame
        Columns: ['Wavelength', 'Red', 'Green', 'Blue']
        Values in photons/(s·µm²·nm).
    """
    h = 6.62607015e-34   # J·s
    c = 2.99792458e8     # m/s

    wl = normalized_df["Wavelength"].values
    lam_m = wl * 1e-9
    E_photon = (h * c) / lam_m  # J per photon

    out = {"Wavelength": wl}

    for ch in ["Red", "Green", "Blue"]:
        spectrum = normalized_df[ch].values if ch in normalized_df.columns else np.zeros_like(wl)
        P = led_powers_watts.get(ch, 0.0)

        if P > 0:
            mask = ~np.isnan(spectrum)
            if mask.sum() > 1:
                # Spectrum-weighted mean photon energy
                avg_E = np.trapezoid(E_photon[mask] * spectrum[mask], wl[mask])
                if avg_E > 0:
                    photons_total = P / avg_E
                    out[ch] = (photons_total / measurement_area_um2) * spectrum
                    continue

        # Zero power or bad data → zero flux
        out[ch] = np.zeros_like(wl, dtype=float)

    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────
# 4. Load power measurements
# ─────────────────────────────────────────────────────────────────────
def load_power_csv(
    file_path: Path = POWER_CSV_PATH,
    row_label: str | None = None,
) -> dict:
    """
    Load per-channel power measurements from AOSLO_power.csv.

    Expected CSV format:
        Date,Red,Green,Blue,nW for 250 um spot
        9/25/23,0,1870,0,

    Parameters
    ----------
    file_path : Path
        Path to the AOSLO power CSV.
    row_label : str or None
        Value in the 'Date' column to select.  If None, uses the first
        data row.

    Returns
    -------
    dict
        {'Red': watts, 'Green': watts, 'Blue': watts}
        Zero values are preserved (not omitted).
    """
    df = pd.read_csv(file_path)
    df = df.dropna(how="all")

    if row_label is not None:
        row = df[df["Date"].astype(str).str.strip() == str(row_label).strip()]
        if row.empty:
            raise KeyError(
                f"Row '{row_label}' not found in {file_path}. "
                f"Available: {df['Date'].tolist()}"
            )
        row = row.iloc[0]
    else:
        row = df.iloc[0]

    powers = {}
    for ch in ["Red", "Green", "Blue"]:
        if ch in df.columns:
            val = row[ch]
            powers[ch] = float(val) * 1e-9 if pd.notna(val) else 0.0  # nW → W
        else:
            powers[ch] = 0.0

    return powers


# ─────────────────────────────────────────────────────────────────────
# 5. Top-level pipeline
# ─────────────────────────────────────────────────────────────────────
def process_aoslo_data(
    spectra_csv: Path = SPECTRA_PATH,
    power_csv: Path = POWER_CSV_PATH,
    power_row: str | None = None,
    led_powers_watts: dict | None = None,
    measurement_area_um2: float = 169.0 ** 2, # µm^2
) -> pd.DataFrame:
    """
    Full AOSLO processing pipeline.

    1. Load spectroradiometer spectra (3 channels, arbitrary units)
    2. Normalize each channel to unit area
    3. Load power measurements (or use provided overrides)
    4. Convert to photon flux

    Parameters
    ----------
    spectra_csv : Path
        Spectroradiometer CSV (3 channels).
    power_csv : Path
        Power CSV with per-channel nW values.
    power_row : str or None
        Which row (by Date label) to use from the power CSV.
        None = first row.
    led_powers_watts : dict or None
        If provided, {'Red': watts, 'Green': watts, 'Blue': watts}
        overrides the power CSV values.
    measurement_area_um2 : float
        Area over which the power meter measured total power (µm²).
        Default: 169**2 µm^2 (1.7 degree AOSLO full-field rectangle ON THE RETINA, where 1 degree is equal to 0.288mm according to Kolb and Marshak 2003).

    Returns
    -------
    pd.DataFrame
        Columns: ['Wavelength', 'Red', 'Green', 'Blue']
        Photon flux in photons/(s·µm²·nm).
        Channels with zero power have all-zero values.
    """
    # 1 – load spectra
    spectra_df = load_aoslo_spectra(spectra_csv)

    # 2 – normalize
    norm_df = normalize_spectra(spectra_df)

    # 3 – get power values
    if led_powers_watts is None:
        led_powers_watts = load_power_csv(power_csv, row_label=power_row)

    # 4 – convert to photon flux
    photon_flux_df = convert_to_photon_flux(
        norm_df,
        led_powers_watts=led_powers_watts,
        measurement_area_um2=measurement_area_um2,
    )

    return photon_flux_df
