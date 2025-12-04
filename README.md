# OpsinActivation

OpsinActivation is a GUI-driven tool designed to model and visualize photoreceptor activation under custom illumination conditions. It is built around a workflow tailored for retinal research, allowing users to configure species, select LED configurations, generate activation plots, and compute LED power ratios.

---

## Features

### **Spectral Activation Modeling**

* Computes opsin activation based on uploaded spectral sensitivity data.
* Supports multiple species and opsin sets.
* Integrates LED spectral profiles to estimate photoreceptor-specific activation.

### **LED Configuration Controls**

* Users may choose from predefined LED triplets (e.g., RGB, RGUV, GBUV).
* Recommends relative LED powers (in nW) for target activation goals.
* Allows matrix export for downstream analysis.

### **Interactive Plotting**

* Generates line plots with shaded scotopic, mesopic, and photopic zones.
* Highlights LED vector contributions and activation levels.
* Allows visual inspection of data point labels, axis scaling, and shading.

---

## Repository Structure

```
/mnt/data
    ├── activation.py      # Core math for opsin activation calculations
    ├── plotting.py        # Plot generation utilities
    ├── plots.py           # High-level plot management for GUI
    ├── nomograms.py       # Functions related to opsin sensitivity curves
    ├── io_led.py          # I/O routines for LED spectral data
    ├── controls.py        # PyQt6 widget for user input controls
    ├── app.py             # Main GUI application entry point
    └── __init__.py        # Package initialization
```

---

## Installation

1. **Clone the repository**
2. **Install dependencies** (Python 3.10+ recommended):

   ```
   pip install -r requirements.txt
   ```
3. **Run the application**:

   ```
   python app.py
   ```

---

## Usage Overview

1. **Launch the GUI.**
2. **Select species** from the dropdown.
3. **Choose LED configuration** (e.g., RGB or RGUV).
4. **Import or use default LED spectra.**
5. **Adjust parameters** and generate activation or LED isolation plots.
6. **Use the “Recommend LED Ratios” button** to compute suggested LED power levels in nW.
7. **Export matrices or figures** as needed.

---

## LED Power Calibration Files

OpsinActivation loads LED power calibration CSV files from:

```
OpsinActivationGUI/assets/
```

Two example calibrations are included:

* `Hyperscope_LED_power.csv`
* `MP2000_LED_power.csv`

A **Scope** dropdown in the GUI selects which file to use.

### CSV Format

Each calibration file must follow this structure:

```csv
Date,Red,Green,Blue,UV,nW for 250 um spot
Current,1430,168,259,214,
2025-04-30 to 2025-07-19,2856,242.1,813,887,
...
```

* **Date** — label shown in the GUI
* **Red/Green/Blue/UV** — measured LED power in **nW** for a 250-µm spot
* The final column is ignored (notes only)

Add new calibrations by simply adding new rows.

---

## Creating Your Own Calibration File

To build a custom calibration for your own microscope or LED system, you must measure:

1. **LED emission spectra**
2. **Filter transmission spectra** (excitation filters, dichroics)
3. **Total LED power at the sample plane** (Watts or µW)
4. The illumination **spot diameter** (default: 250 µm)

OpsinActivation uses LED spectra to model wavelength-dependent activation, and your CSV supplies the *magnitude* of each LED’s output.

### Minimum CSV Requirements

You only need total optical power (in nW) for each LED channel:

```csv
Date,Red,Green,Blue,UV
MyScope-2025,520,130,310,410
```

Save it into:

```
OpsinActivationGUI/assets/
```

and select it using the GUI Scope dropdown.

---

## Power Meter Correction

OpsinActivation compensates for the spectral responsivity of your powermeter.

Many powermeters measure green light more efficiently than UV, which would otherwise distort relative LED calibration.

### How Correction Works

The pipeline:

1. Loads your powermeter responsivity curve
2. Computes per-wavelength correction factors
3. Applies them to the LED spectra before normalization

This yields more accurate photon-flux estimates.

### Disabling the Correction

If you prefer to use raw LED spectra, you may:

#### **Option A — Set all correction values to 1**

Edit your powermeter CSV so that responsivity is constant. OpsinActivation will compute correction curves of all ones.

#### **Option B — Hard-disable in code**

Replace in `io_led.py`:

```python
corr = calculate_powermeter_corrections(powermeter_csv, wavelengths)
```

with:

```python
corr = pd.DataFrame({
    "Wavelength": wavelengths,
    "Red_Correction": 1,
    "Green_Correction": 1,
    "Blue_Correction": 1,
    "UV_Correction": 1,
})
```

---

## Roadmap

* Parameter presets for common lab species
* Extended plotting customization
* Batch processing options
* Improved wavelength-dependent visualization

---

## License

This project currently does not specify a license. Add one if distribution is intended.

---

## Contact

For questions or contributions, please open an issue or submit a pull request in the project repository.
