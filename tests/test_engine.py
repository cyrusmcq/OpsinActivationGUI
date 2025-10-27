from opsinlab.io_led import (
    LED_SPECTRA_PATH, POWERMETER_PATH,
    load_led_and_filter_spectra, calculate_powermeter_corrections
)

def test_asset_paths_exist():
    assert LED_SPECTRA_PATH.exists(), f"Missing {LED_SPECTRA_PATH}"
    assert POWERMETER_PATH.exists(), f"Missing {POWERMETER_PATH}"

def test_can_load_led_and_filters():
    wavelengths, eff = load_led_and_filter_spectra()
    assert len(wavelengths) > 0
    assert "Wavelength" in eff.columns
    for col in ["Red", "Green", "Blue", "UV"]:
        assert col in eff.columns, f"Missing LED column {col}"

def test_can_build_powermeter_corrections():
    wavelengths, _ = load_led_and_filter_spectra()
    corr = calculate_powermeter_corrections(target_wavelengths=wavelengths)
    assert len(corr) == len(wavelengths)
    for col in ["Red_Correction", "Green_Correction", "Blue_Correction", "UV_Correction"]:
        assert col in corr.columns
