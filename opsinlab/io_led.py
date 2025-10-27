"""
opsinlab.io_led
---------------
Functions for loading LED spectra, applying power-meter corrections,
normalizing spectra, and converting to photon flux.

Assets used:
  - assets/LED_spectra.csv
  - assets/powermeter.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Default: project_root/assets where project_root is two levels above this file
# (opsinlab/ -> project root)
_DEFAULT_ASSETS = (Path(__file__).resolve().parent.parent / "assets").resolve()

# Optional override via env var (handy if you move assets or run from elsewhere)
_ENV_OVERRIDE = os.environ.get("OPSIN_ASSETS_DIR")
ASSETS_DIR = Path(_ENV_OVERRIDE).resolve() if _ENV_OVERRIDE else _DEFAULT_ASSETS

LED_SPECTRA_PATH = ASSETS_DIR / "LED_spectra.csv"
POWERMETER_PATH  = ASSETS_DIR / "powermeter.csv"

def set_assets_dir(path: str | os.PathLike) -> None:
    """
    Optionally set/override the assets directory at runtime (e.g., in tests).
    """
    global ASSETS_DIR, LED_SPECTRA_PATH, POWERMETER_PATH
    ASSETS_DIR = Path(path).resolve()
    LED_SPECTRA_PATH = ASSETS_DIR / "LED_spectra.csv"
    POWERMETER_PATH  = ASSETS_DIR / "powermeter.csv"

# ---------------------------------------------------------------------
# 1. Load LED and filter spectra
# ---------------------------------------------------------------------
def load_led_and_filter_spectra(file_path: Path = LED_SPECTRA_PATH):
    """
    Load LED spectra and filter transmission, returning wavelength-aligned effective spectra.

    Parameters
    ----------
    file_path : Path
        CSV path containing LED and filter data (default = assets/LED_spectra.csv).

    Returns
    -------
    wavelengths : np.ndarray
    effective_spectra : pd.DataFrame
        Columns: ['Red', 'Green', 'Blue', 'UV']
    """
    data = pd.read_csv(file_path, skiprows=3)
    wavelengths = data.iloc[:, 0].astype(float).values

    led_spectra = data.iloc[:, 1:5].astype(float)
    led_spectra.columns = ["Red", "Green", "Blue", "UV"]

    filter_spectra = data.iloc[:, 6:10].astype(float)
    filter_spectra.columns = ["Red", "Green", "Blue", "UV"]

    effective_spectra = led_spectra * filter_spectra
    effective_spectra.insert(0, "Wavelength", wavelengths)

    return wavelengths, effective_spectra


# ---------------------------------------------------------------------
# 2. Power meter correction
# ---------------------------------------------------------------------
def calculate_powermeter_corrections(
    file_path: Path = POWERMETER_PATH,
    target_wavelengths: np.ndarray | None = None
):
    """
    Interpolate power-meter responsivity and compute per-LED correction curves.

    Parameters
    ----------
    file_path : Path
        Path to powermeter CSV (default = assets/powermeter.csv)
    target_wavelengths : np.ndarray
        Wavelengths to interpolate to (e.g., from LED spectra)

    Returns
    -------
    correction_table : pd.DataFrame
        Columns: ['Wavelength', 'Red_Correction', 'Green_Correction', 'Blue_Correction', 'UV_Correction']
    """
    sens = pd.read_csv(file_path, skiprows=3)
    wavelengths = sens.iloc[:, 1].dropna().astype(float)
    responsivity = sens.iloc[:, 2].dropna().astype(float)

    sens = pd.DataFrame({"Wavelength": wavelengths, "Responsivity": responsivity}).dropna().sort_values("Wavelength")

    if target_wavelengths is None:
        target_wavelengths = sens["Wavelength"].values

    interp_resp = np.interp(target_wavelengths, sens["Wavelength"], sens["Responsivity"])

    ref_wavelengths = {"Red": 640, "Green": 550, "Blue": 465, "UV": 390}
    correction_table = pd.DataFrame({"Wavelength": target_wavelengths})

    for led, ref_wl in ref_wavelengths.items():
        r_ref = np.interp(ref_wl, sens["Wavelength"], sens["Responsivity"])
        correction_table[f"{led}_Correction"] = 1 / (interp_resp / r_ref)

    return correction_table


# ---------------------------------------------------------------------
# 3. Apply corrections
# ---------------------------------------------------------------------
def apply_power_meter_corrections(effective_df: pd.DataFrame, correction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiply effective LED spectra by power-meter correction curves.
    Returns ONLY: ['Wavelength', 'Red_Corrected', 'Green_Corrected', 'Blue_Corrected', 'UV_Corrected']
    """
    led_cols = ["Red", "Green", "Blue", "UV"]
    out = {"Wavelength": effective_df["Wavelength"].values}
    merged = pd.merge(
        effective_df[["Wavelength"] + led_cols],
        correction_df[["Wavelength", "Red_Correction", "Green_Correction", "Blue_Correction", "UV_Correction"]],
        on="Wavelength",
        how="inner",
    )
    for led in led_cols:
        out[f"{led}_Corrected"] = merged[led] * merged[f"{led}_Correction"]
    return pd.DataFrame(out)

# ---------------------------------------------------------------------
# 4. Normalize spectra (area = 1)
# ---------------------------------------------------------------------
def normalize_spectra(corrected_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize corrected spectra by trapezoid area.
    Returns ONLY: ['Wavelength', 'Red_Corrected_Normalized', ...]
    """
    wl = corrected_df["Wavelength"].values
    out = {"Wavelength": wl}
    corr_cols = [c for c in corrected_df.columns if c.endswith("_Corrected")]
    for col in corr_cols:
        vals = corrected_df[col].values
        mask = ~np.isnan(vals)
        if mask.sum() > 1:
            area = np.trapezoid(vals[mask], wl[mask])
            out[f"{col}_Normalized"] = (vals / area) if area > 0 else vals
        else:
            out[f"{col}_Normalized"] = vals
    return pd.DataFrame(out)

# ---------------------------------------------------------------------
# 5. Convert to photon flux (photons/s/µm²)
# ---------------------------------------------------------------------
def convert_to_photon_flux(normalized_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert normalized spectra to photon flux using measured LED powers.
    Returns ONLY: ['Wavelength', 'Red_PhotonFlux', ...]
    """
    h = 6.62607015e-34
    c = 2.99792458e8
    led_powers = {"Red": 1.43e-6, "Green": 1.68e-7, "Blue": 2.59e-7, "UV": 2.140e-7}
    wl = normalized_df["Wavelength"].values
    lam_m = wl * 1e-9
    E = (h * c) / lam_m

    out = {"Wavelength": wl}
    norm_cols = [c for c in normalized_df.columns if c.endswith("_Normalized")]
    for col in norm_cols:
        led = col.split("_")[0]
        spectrum = normalized_df[col].values
        if led not in led_powers:
            continue
        P = led_powers[led]
        mask = ~np.isnan(spectrum)
        if mask.sum() > 1:
            avg_E = np.trapezoid(E[mask] * spectrum[mask], wl[mask])
            if avg_E > 0:
                spot_area = np.pi * (250 ** 2)  # µm²
                photons_total = P / avg_E
                out[f"{led}_PhotonFlux"] = (photons_total / spot_area) * spectrum
                continue
        # fallback (no valid avg_E)
        out[f"{led}_PhotonFlux"] = np.full_like(wl, np.nan, dtype=float)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------
# 6. Convenience pipeline
# ---------------------------------------------------------------------
def process_led_data(
    led_csv: Path = LED_SPECTRA_PATH,
    powermeter_csv: Path = POWERMETER_PATH
) -> pd.DataFrame:
    """
    End-to-end LED processing pipeline returning photon flux spectra.
    """
    wavelengths, eff = load_led_and_filter_spectra(led_csv)
    corr = calculate_powermeter_corrections(powermeter_csv, wavelengths)
    corr_applied = apply_power_meter_corrections(eff, corr)
    norm = normalize_spectra(corr_applied)
    photon_flux = convert_to_photon_flux(norm)
    return photon_flux
# ---------------------------------------------------------------------
# 7. Save-all pipeline (for quick verification)
# ---------------------------------------------------------------------
from datetime import datetime

def save_all_outputs(
    output_dir: str | Path = "outputs",
    led_csv: Path | None = None,
    powermeter_csv: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run the full pipeline and save each stage to CSV in `output_dir`.

    Returns a dict of DataFrames:
        {
          "effective": ...,
          "corrections": ...,
          "corrected": ...,
          "normalized": ...,
          "photon_flux": ...
        }
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if led_csv is None:
        led_csv = LED_SPECTRA_PATH
    if powermeter_csv is None:
        powermeter_csv = POWERMETER_PATH

    # 1) Load LED*filter
    wavelengths, effective = load_led_and_filter_spectra(led_csv)
    effective.to_csv(out / "EffectiveSpectra.csv", index=False)

    # 2) Corrections
    corrections = calculate_powermeter_corrections(powermeter_csv, wavelengths)
    corrections.to_csv(out / "PowerMeterCorrectionCurves.csv", index=False)

    # 3) Apply corrections
    corrected = apply_power_meter_corrections(effective, corrections)
    corrected.to_csv(out / "CorrectedSpectra.csv", index=False)

    # 4) Normalize
    normalized = normalize_spectra(corrected)
    normalized.to_csv(out / "NormalizedSpectra.csv", index=False)

    # 5) Photon flux
    photon_flux = convert_to_photon_flux(normalized)
    photon_flux.to_csv(out / "PhotonFluxSpectra.csv", index=False)

    # Small metadata breadcrumb
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "assets_dir": str(ASSETS_DIR),
        "led_csv": str(led_csv),
        "powermeter_csv": str(powermeter_csv),
        "output_dir": str(out),
    }
    pd.Series(meta).to_csv(out / "run_metadata.txt")

    return {
        "effective": effective,
        "corrections": corrections,
        "corrected": corrected,
        "normalized": normalized,
        "photon_flux": photon_flux,
    }


# Optional CLI: `python -m opsinlab.io_led`
if __name__ == "__main__":
    dfs = save_all_outputs()
    print("Wrote outputs to:", (Path("outputs").resolve()))
    for name, df in dfs.items():
        print(f"  {name}: shape={df.shape}")