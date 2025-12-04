from PyQt6 import QtWidgets, QtCore
from opsinlab.io_led import process_led_data
from opsinlab.nomograms import generate_nomograms, available_species, interpolate_nomograms
from opsinlab.activation import isolation_pipeline, compute_midgray_and_amplitude, recommend_led_power_ratios
from gui.widgets.controls import Controls
from gui.widgets.plots import Plots, _led_set_code
from opsinlab.activation import opsin_weighted_isomerizations

import numpy as np
import os
from datetime import datetime
import pandas as pd
import math


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpsinActivation")
        self._saving = False

        # central layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # left controls, right plots
        self.controls = Controls(
            species_list=available_species(),
            led_configs=["RGB", "RGUV", "GBUV"],
        )

        self.plots = Plots()
        layout.addWidget(self.controls, 0)
        layout.addWidget(self.plots, 1)

        # --- load initial power sets from CSV based on scope ---
        initial_scope = self.controls.scope_combo.currentText()
        self._POWER_SETS_NW = self._load_power_csv(initial_scope)

        # push the keys into Controls
        self.controls.power_combo.clear()
        self.controls.power_combo.addItems(list(self._POWER_SETS_NW.keys()))

        def _powers_watts_from_key(key: str) -> dict:
            # Fallback-friendly: if control not present or key missing, use Current
            if not key:
                key = "Current"
            sel = self._POWER_SETS_NW.get(key, self._POWER_SETS_NW["Current"])
            return {k: v * 1e-9 for k, v in sel.items()}  # nW -> W

        self._powers_watts_from_key = _powers_watts_from_key

        # --- Cache photon flux BASE using the current dropdown selection (if present) ---
        try:
            power_key = self.controls.current_power_key()
        except Exception:
            power_key = "Current"

        self._photon_flux_base = process_led_data(
            led_powers_watts=self._powers_watts_from_key(power_key)
        )
        self._photon_flux = self._photon_flux_base.copy()  # working (attenuated) copy

        # last outputs cache for saving
        self._last_out = None

        # re-entrancy guard to prevent signal loops
        self._in_recompute = False

        # wire events
        self.controls.paramsChanged.connect(self.recompute)

        # scope-specific reload
        self.controls.scope_combo.currentIndexChanged.connect(self._scope_changed)

        # ensure single connection just like saveMatricesClicked
        try:
            self.controls.recommendRatiosClicked.disconnect(self._recommend_led_ratios)
        except (TypeError, AttributeError):
            pass
        self.controls.recommendRatiosClicked.connect(self._recommend_led_ratios)

        # ensure single connection
        try:
            self.controls.saveMatricesClicked.disconnect(self._save_matrices)
        except TypeError:
            pass
        self.controls.saveMatricesClicked.connect(self._save_matrices)

        # status bar for quick summaries
        self.statusBar().showMessage("Ready")

        # initial compute
        QtCore.QTimer.singleShot(0, self.recompute)

    # ---------- helpers ----------
    def _load_power_csv(self, scope_name: str) -> dict:
        """
        Load LED power calibration sets (in nW) for the given scope.

        Expected CSV format (your files):
            Date,Red,Green,Blue,UV,nW for 250 um spot

        We use the 'Date' column as the key (e.g. 'Current',
        '2025-04-30 to 2025-07-19', etc.) and store a dict of
        {LED_name: power_nW}.
        """

        if scope_name == "Hyperscope":
            filename = "Hyperscope_LED_power.csv"
        else:
            filename = "MP2000_LED_power.csv"

        # app.py is in gui/, assets is at OpsinActivationGUI/assets
        project_root = os.path.dirname(os.path.dirname(__file__))
        assets_dir = os.path.join(project_root, "assets")
        path = os.path.join(assets_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find power CSV at: {path}")

        df = pd.read_csv(path)

        # Drop rows that are completely empty (handles trailing blank lines)
        df = df.dropna(how="all")

        if "Date" not in df.columns:
            raise ValueError(f"CSV {path} must contain a 'Date' column; found: {df.columns.tolist()}")

        power_sets: dict[str, dict[str, float]] = {}

        for _, row in df.iterrows():
            # Use 'Date' as the label (Current, test, date ranges, etc.)
            raw_name = row["Date"]
            if pd.isna(raw_name):
                continue
            name = str(raw_name).strip()
            if not name:
                continue

            # Build LED -> nW dict; ignore the annotation column
            d: dict[str, float] = {}
            for col in df.columns:
                if col in ("Date", "nW for 250 um spot"):
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                d[col] = float(val)

            if d:
                power_sets[name] = d

        if not power_sets:
            raise ValueError(f"No usable power sets parsed from {path}")

        return power_sets


    def _scope_changed(self, *_):
        """Handle changes in scope (Hyperscope vs MP2000)."""
        scope = self.controls.current_scope()
        try:
            self._POWER_SETS_NW = self._load_power_csv(scope)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Failed to load power sets", str(e))
            return

        # Update the calibration dropdown without causing extra recomputes
        self.controls.power_combo.blockSignals(True)
        self.controls.power_combo.clear()
        self.controls.power_combo.addItems(list(self._POWER_SETS_NW.keys()))
        self.controls.power_combo.blockSignals(False)

        # Recompute with the new calibration
        self.recompute()

    @staticmethod
    def _format_iso_summary(iso_rates: dict) -> str:
        """Build a concise status-bar string from opsin_weighted_isomerizations() output."""
        if not iso_rates:
            return ""
        vals = iso_rates.get("Rstar_per_opsin") or {}
        if not isinstance(vals, dict) or not vals:
            return ""
        # stable opsin order if present
        order = ["Rh", "LW", "MW", "SW", "UVS"]
        keys = [k for k in order if k in vals] + [k for k in vals.keys() if k not in order]
        parts = [f"{k}: {vals[k]:.3e} R*/s" for k in keys]
        return " | ".join(parts)

    # ---------- core ----------
    def recompute(self):
        if self._in_recompute:
            return
        self._in_recompute = True
        try:
            species = self.controls.current_species()
            leds = self.controls.selected_leds()

            # === Refresh photon-flux BASE from the selected power set ===
            try:
                power_key = self.controls.current_power_key()
            except Exception:
                power_key = "Current"

            self._photon_flux_base = process_led_data(
                led_powers_watts=self._powers_watts_from_key(power_key)
            )

            # --- ND + PMT attenuation ---
            try:
                od_val = self.controls.nd_optical_density()
            except Exception:
                od_val = 0.0
            try:
                pmt_on = self.controls.pmt_lever_enabled()
            except Exception:
                pmt_on = False

            # ND transmission (log units); PMT lever (70/30 mirror)
            try:
                T_nd = 10 ** (-(float(od_val) if od_val else 0.0))
            except Exception:
                T_nd = 1.0
            T_pmt = 0.3 if pmt_on else 1.0
            transmission = T_nd * T_pmt

            # Apply attenuation to a pristine copy
            pf = self._photon_flux_base.copy()
            for col in pf.columns:
                if col.endswith("_PhotonFlux") or col in ("Red", "Green", "Blue", "UV"):
                    pf[col] = pf[col] * transmission
            self._photon_flux = pf

            # LED grid (e.g., 300–700 from your photon-flux table)
            wl_led = pf["Wavelength"].to_numpy(dtype=float)
            wl_master = np.arange(200.0, 701.0, 1.0)  # 200–700 nm, 1-nm steps

            # Build master 200–700 nm grid and normalize area over 200–700
            nomo_200_700 = generate_nomograms(
                species,
                wl_master,
                normalize=True,  # area = 1 over 200–700
            )

            if hasattr(self.plots, "update_nomogram_plot"):
                self.plots.update_nomogram_plot(nomo_200_700)

            # Interpolate back to the LED grid for all math (alignment guaranteed)
            nomo = interpolate_nomograms(nomo_200_700, wl_led)

            # Isolation settings
            strategy = "match_across_isolations" if self.controls.match_across_isolations() else "per_isolation_max"
            use_inv = self.controls.use_exact_inverse()

            # --- Core computation ---
            out = isolation_pipeline(
                photon_flux_df=pf,
                nomograms_df=nomo,
                species=species,
                selected_leds_3=leds,
                strategy=strategy,
                use_exact_inverse=use_inv,
            )

            # --- Opsin-weighted R* (uses attenuated pf and selected LEDs) ---
            iso_rates = opsin_weighted_isomerizations(
                photon_flux_df=pf,
                nomograms_df=nomo,
                species=species,
                selected_leds_3=leds,
                quantum_efficiency=0.67,
            )
            print("[R*/s per opsin]", {k: f"{v:.2e}" for k, v in iso_rates["Rstar_per_opsin"].items()})

            # Extract per-opsin R*/s and a single rod R* value for the summary tab
            rstar_dict = iso_rates.get("Rstar_per_opsin") or {}
            opsin_rstars = dict(rstar_dict) if isinstance(rstar_dict, dict) else {}

            rod_rstar = None
            # Try common rod labels in order
            for key in ("Rh", "Rho", "Rod"):
                if key in rstar_dict:
                    try:
                        rod_rstar = float(rstar_dict[key])
                    except Exception:
                        rod_rstar = None
                    break

            # cache & enable save button
            self._last_out = out | {"iso_rates": iso_rates}
            self.controls.save_btn.setEnabled(True)

            # update plots once
            A = out["ActivationMatrix"]
            mods = out["Modulations"]

            # --- mid-gray + amplitude ---
            try:
                mid = compute_midgray_and_amplitude(mods)
                gamma_global = float(mid["gamma_max"].min()) if not mid.empty else None
            except Exception:
                mid = None
                gamma_global = None

            self.plots.update_all(
                photon_flux_df=pf,
                nomograms_df=nomo,
                activation_matrix=A,
                modulations_df=mods,
                selected_leds=leds,
                species=species,
            )

            # --- Build isolating LED vectors for the Summary tab ---
            iso_vectors = {}
            if mods is not None and not mods.empty:
                leds_all = list(mods.index)
                leds3 = [L for L in leds if L in leds_all] or leds_all
                code = _led_set_code(leds3)  # e.g. "RGB", "RGUV", "GBUV"

                for col in mods.columns:
                    try:
                        vec = [float(mods.loc[L, col]) for L in leds3]
                    except Exception:
                        continue
                    ops_name = col.replace("_isolating", "")
                    label = f"{ops_name} ({code})"
                    iso_vectors[label] = vec

            # push mid-gray readout
            if hasattr(self.plots, "set_midgray"):
                self.plots.set_midgray(mid, gamma_global)

            # --- Update Summary tab (if present) ---
            if hasattr(self.plots, "update_summary"):
                self.plots.update_summary(
                    rod_rstar=rod_rstar,
                    opsin_rstars=opsin_rstars,
                    photon_flux_df=pf,
                    selected_leds=leds,
                    iso_vectors=iso_vectors,
                )

            # also echo to status bar so you always see a readout
            sb_text = self._format_iso_summary(iso_rates)
            if sb_text:
                self.statusBar().showMessage(f"Isomerizations — {sb_text}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error in recompute()", str(e))
            raise
        finally:
            self._in_recompute = False

    def _save_matrices(self):
        # (unchanged – your existing implementation)
        ...
        # keep your existing _save_matrices body here

    def _recommend_led_ratios(self):
        # (unchanged – your existing implementation)
        ...
        # keep your existing _recommend_led_ratios body here


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 760)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
