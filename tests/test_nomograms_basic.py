import numpy as np
import pandas as pd
import pytest
from opsinlab.nomograms import (
    generate_nomograms,
    species_data,
    normalize_curve,
)

@pytest.mark.parametrize("species_name", list(species_data.keys()))
def test_nomogram_area_and_peak(species_name):
    # === wavelength grid ===
    wavelengths = np.arange(300, 701, 1)

    # === generate normalized nomograms ===
    df = generate_nomograms(species_name, wavelengths, normalize=True)

    for opsin, lmax in species_data[species_name].items():
        curve = df[opsin].values

        # --- (1) area normalization check ---
        mask = np.isfinite(curve)
        area = np.trapezoid(curve[mask], wavelengths[mask])
        assert np.isclose(area, 1.0, rtol=1e-2), (
            f"{species_name} {opsin}: area={area:.3f} not close to 1"
        )

        # --- (2) peak location near λmax ---
        peak_wl = wavelengths[np.nanargmax(curve)]
        diff = abs(peak_wl - lmax)
        assert diff < 10, (
            f"{species_name} {opsin}: peak {peak_wl:.1f} nm differs from λmax {lmax:.1f} nm by {diff:.1f}"
        )

def test_normalize_curve_is_idempotent():
    # a curve already normalized should remain normalized
    wl = np.linspace(300, 700, 401)
    curve = np.exp(-0.5 * ((wl - 500) / 30) ** 2)
    from opsinlab.nomograms import normalize_curve
    norm1 = normalize_curve(curve, wl)
    norm2 = normalize_curve(norm1, wl)
    area1 = np.trapezoid(norm1, wl)
    area2 = np.trapezoid(norm2, wl)
    assert np.isclose(area1, 1.0, rtol=1e-3)
    assert np.isclose(area2, 1.0, rtol=1e-3)
