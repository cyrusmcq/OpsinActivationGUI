"""
opsinlab.io_oled
----------------
Functions for loading Rig1 OLED spectra, calibrating with power-meter
measurements at multiple wavelengths, and converting to photon flux.

Assets used:
  - assets/OLED_spectra.csv       (spectroradiometer, arbitrary units)
  - assets/OLED_power.csv         (power-meter readings at calibration λs)

Pipeline
--------
1. load_oled_spectrum()       – load spectroradiometer data (arb. units)
2. load_power_calibration()   – load power-meter readings & calibration λs
3. calibrate_spectrum()       – anchor arb-unit spectrum to absolute W/nm
                                using power-meter readings
4. convert_to_photon_flux()   – convert spectral power density to
                                photons/(s·µm²) for a given spot size
5. process_oled_data()        – top-level convenience function
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# ── Asset paths ──────────────────────────────────────────────────────
_DEFAULT_ASSETS = (Path(__file__).resolve().parent.parent / "assets").resolve()
_ENV_OVERRIDE = os.environ.get("OPSIN_ASSETS_DIR")
ASSETS_DIR = Path(_ENV_OVERRIDE).resolve() if _ENV_OVERRIDE else _DEFAULT_ASSETS

LED_SPECTRA_PATH = ASSETS_DIR / "OLED_spectra.csv"
POWER_CSV_PATH = ASSETS_DIR / "Rig1_OLED_power.csv"


def set_assets_dir(path: str | os.PathLike) -> None:
    """Optionally set/override the assets directory at runtime."""
    global ASSETS_DIR, LED_SPECTRA_PATH, POWER_CSV_PATH
    ASSETS_DIR = Path(path).resolve()
    LED_SPECTRA_PATH = ASSETS_DIR / "OLED_spectra.csv"
    POWER_CSV_PATH = ASSETS_DIR / "Rig1_OLED_power.csv"


# ─────────────────────────────────────────────────────────────────────
# 1. Load spectroradiometer spectrum
# ─────────────────────────────────────────────────────────────────────
def load_oled_spectrum(file_path: Path = LED_SPECTRA_PATH) -> pd.DataFrame:
    data = pd.read_csv(file_path, skiprows=3)
    wavelengths = data.iloc[:, 0].astype(float).values
    intensity = data.iloc[:, 1].astype(float).values

    df = pd.DataFrame({
        "Wavelength": wavelengths,
        "White_arb": intensity,
    })
    df = df.dropna(subset=["Wavelength", "White_arb"])
    return df

# ─────────────────────────────────────────────────────────────────────
# 2. Load power-meter calibration data
# ─────────────────────────────────────────────────────────────────────
def load_power_calibration(
    file_path: Path = POWER_CSV_PATH,
    row_label: str | None = None,
) -> dict:
    """
    Load power-meter readings from OLED_power.csv.

    Expected CSV format (tab-separated):
        Date    543     600     680     nW for 250 um spot
        7/18/24 366     361.3   271.5

    Parameters
    ----------
    file_path : Path
        Path to the OLED power CSV.
    row_label : str or None
        Value in the 'Date' column to select.  If None, uses the first
        data row.

    Returns
    -------
    dict
        Keys are calibration wavelengths (float, nm), values are
        measured power in watts.
        Example: {543.0: 3.66e-7, 600.0: 3.613e-7, 680.0: 2.715e-7}
    """
    df = pd.read_csv(file_path)

    if row_label is not None:
        row = df[df.iloc[:, 0].astype(str) == str(row_label)]
        if row.empty:
            raise KeyError(
                f"Row '{row_label}' not found in {file_path}. "
                f"Available: {df.iloc[:, 0].tolist()}"
            )
        row = row.iloc[0]
    else:
        row = df.iloc[0]

    # Columns whose headers are numeric wavelengths
    cal_points = {}
    for col in df.columns:
        try:
            wl = float(col)
        except ValueError:
            continue
        val = float(row[col])
        cal_points[wl] = val * 1e-9  # nW → W

    if not cal_points:
        raise ValueError(
            f"No numeric wavelength columns found in {file_path}. "
            f"Columns: {df.columns.tolist()}"
        )

    return cal_points


# ─────────────────────────────────────────────────────────────────────
# 3. Calibrate spectrum to absolute spectral power density (W/nm)
# ─────────────────────────────────────────────────────────────────────
def calibrate_spectrum(
    spectrum_df: pd.DataFrame,
    cal_points: dict,
) -> pd.DataFrame:
    """
    Convert an arbitrary-unit spectrum to absolute spectral power
    density (W/nm) using power-meter calibration points.

    For each calibration wavelength λ_cal, the power meter reports
    the *total* power of the broadband source (with its sensitivity
    correction set to λ_cal).  The true total power is:

        P_total = ∫ S_arb(λ) dλ  ×  scale_factor

    Each calibration reading gives an independent estimate of
    scale_factor.  We average them (or could flag outliers).

    Parameters
    ----------
    spectrum_df : pd.DataFrame
        Must contain 'Wavelength' and 'White_arb' columns.
    cal_points : dict
        {wavelength_nm: power_watts} from the power meter.

    Returns
    -------
    pd.DataFrame
        Columns: ['Wavelength', 'White_W_per_nm']
        The spectrum in absolute units of W/nm.
    """
    wl = spectrum_df["Wavelength"].values
    s_arb = spectrum_df["White_arb"].values

    # Total area under the arbitrary-unit spectrum
    total_area = np.trapezoid(s_arb, wl)
    if total_area <= 0:
        raise ValueError("Spectrum has non-positive integral; cannot calibrate.")

    # Each power-meter reading is an estimate of total power,
    # so each gives:  scale_factor = P_measured / total_area
    scale_factors = {}
    for wl_cal, p_measured in cal_points.items():
        scale_factors[wl_cal] = p_measured / total_area

    # Report per-wavelength scale factors for diagnostics
    sf_values = np.array(list(scale_factors.values()))
    sf_mean = sf_values.mean()
    sf_std = sf_values.std()
    cv = (sf_std / sf_mean * 100) if sf_mean > 0 else 0

    if cv > 20:
        print(
            f"WARNING: Power-meter calibration points show {cv:.1f}% CV "
            f"(scale factors: {scale_factors}).  "
            f"This may indicate a badly calibrated power meter at one "
            f"or more wavelengths."
        )
    else:
        print(
            f"Power-meter calibration: CV = {cv:.1f}% across "
            f"{len(cal_points)} wavelengths (good agreement)."
        )

    # Use the mean scale factor
    spectral_power = s_arb * sf_mean  # W/nm

    return pd.DataFrame({
        "Wavelength": wl,
        "White_W_per_nm": spectral_power,
    })


# ─────────────────────────────────────────────────────────────────────
# 4. Convert spectral power density → photon flux
# ─────────────────────────────────────────────────────────────────────
def convert_to_photon_flux(
    calibrated_df: pd.DataFrame,
    measurement_area_um2: float = np.pi * 600.0 ** 2,
    led_powers_watts: dict | None = None,
) -> pd.DataFrame:
    """
    Convert calibrated spectral power density to photon flux.

    The power meter measures total power over the full illuminated field.
    To get flux (photons/s/µm²), we divide by the measurement area —
    i.e. the area over which the power was actually collected.

    If led_powers_watts is provided (from the GUI power dropdown),
    the spectrum is rescaled so that its total integrated power
    matches the supplied value before computing photon flux.

    Parameters
    ----------
    calibrated_df : pd.DataFrame
        Columns: ['Wavelength', 'White_W_per_nm']
    measurement_area_um2 : float
        Area over which the power meter measured total power (µm²).
        Default: π × 600² (1200 µm diameter spot through 10x objective).
    led_powers_watts : dict or None
        If provided, expects {'White': power_in_watts}.  The spectrum
        is rescaled to this total power before flux conversion.

    Returns
    -------
    pd.DataFrame
        Columns: ['Wavelength', 'White']
        'White' is in photons/(s·µm²·nm) — spectral photon flux density.
        Integrate over wavelength for total photons/(s·µm²).
    """
    h = 6.62607015e-34   # J·s
    c = 2.99792458e8     # m/s

    wl = calibrated_df["Wavelength"].values
    spectral_power = calibrated_df["White_W_per_nm"].values.copy()

    # If a power override is supplied, rescale the spectrum
    if led_powers_watts is not None and "White" in led_powers_watts:
        current_total = np.trapezoid(spectral_power, wl)
        if current_total > 0:
            target_power = led_powers_watts["White"]
            spectral_power = spectral_power * (target_power / current_total)

    # Photon energy at each wavelength: E(λ) = hc/λ
    lam_m = wl * 1e-9
    E_photon = (h * c) / lam_m  # J per photon

    # Spectral photon rate: photons/(s·nm) = P(λ) [W/nm] / E(λ) [J]
    photon_rate = spectral_power / E_photon

    # Divide by measurement area to get flux density
    photon_flux = photon_rate / measurement_area_um2  # photons/(s·µm²·nm)

    return pd.DataFrame({
        "Wavelength": wl,
        "White": photon_flux,
    })


# ─────────────────────────────────────────────────────────────────────
# 5. Top-level pipeline
# ─────────────────────────────────────────────────────────────────────
def process_oled_data(
    led_csv: Path = LED_SPECTRA_PATH,
    power_csv: Path = POWER_CSV_PATH,
    power_row: str | None = None,
    led_powers_watts: dict | None = None,
    measurement_area_um2: float = np.pi * 600.0 ** 2,
) -> pd.DataFrame:
    """
    Full OLED processing pipeline.

    1. Load spectroradiometer spectrum (arbitrary units)
    2. Load power-meter calibration points
    3. Calibrate spectrum to absolute W/nm
    4. Convert to photon flux, optionally rescaling to a GUI-selected
       power level

    Parameters
    ----------
    led_csv : Path
        Spectroradiometer CSV.
    power_csv : Path
        Power-meter CSV with calibration-wavelength columns.
    power_row : str or None
        Which row (by Date label) to use from the power CSV.
        None = first row.
    led_powers_watts : dict or None
        If provided, {'White': watts} to rescale to a specific power.
    measurement_area_um2 : float
        Area over which the power meter measured total power (µm²).
        Default: π × 600² (1200 µm diameter spot through 10x objective) (~1400 µm diameter full field
        through a 10x objective, but there must be a field stop or other occlusion in the path, best to use a smaller spot for power calibration).

    Returns
    -------
    pd.DataFrame
        Columns: ['Wavelength', 'White']
        Photon flux in photons/(s·µm²·nm).
    """
    # 1 – load spectrum
    spectrum_df = load_oled_spectrum(led_csv)

    # 2 – load power-meter calibration
    cal_points = load_power_calibration(power_csv, row_label=power_row)

    # 3 – calibrate to absolute units
    calibrated_df = calibrate_spectrum(spectrum_df, cal_points)

    # 4 – convert to photon flux
    photon_flux_df = convert_to_photon_flux(
        calibrated_df,
        measurement_area_um2=measurement_area_um2,
        led_powers_watts=led_powers_watts,
    )

    return photon_flux_df
