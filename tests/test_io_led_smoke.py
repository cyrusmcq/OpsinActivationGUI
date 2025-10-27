from opsinlab.io_led import (
    LED_SPECTRA_PATH, POWERMETER_PATH,
    load_led_and_filter_spectra, calculate_powermeter_corrections
)

print("LED CSV      :", LED_SPECTRA_PATH, "exists:", LED_SPECTRA_PATH.exists())
print("Powermeter   :", POWERMETER_PATH,   "exists:", POWERMETER_PATH.exists())

wavelengths, eff = load_led_and_filter_spectra()
print("Loaded LED/filter spectra:", eff.shape, "wavelengths:", wavelengths.shape)
print("Columns:", list(eff.columns)[:6], "...")

corr = calculate_powermeter_corrections(target_wavelengths=wavelengths)
print("Loaded corrections:", corr.shape, "columns:", list(corr.columns))

from opsinlab.io_led import process_led_data
from opsinlab.nomograms import generate_nomograms, available_species

# use the same grid the LED pipeline uses
photon_flux = process_led_data()
wl = photon_flux["Wavelength"].values

print("Species:", available_species())

nomo_mouse = generate_nomograms("Mouse", wl, normalize=True)
print(nomo_mouse.head())
