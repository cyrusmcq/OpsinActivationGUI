from PyQt6 import QtWidgets, QtCore
from opsinlab.io_led import process_led_data
from opsinlab.nomograms import generate_nomograms, available_species, interpolate_nomograms
from opsinlab.activation import isolation_pipeline, compute_midgray_and_amplitude, recommend_led_power_ratios
from gui.widgets.controls import Controls
from gui.widgets.plots import Plots
from opsinlab.activation import opsin_weighted_isomerizations
from gui.widgets.plots import Plots, _led_set_code

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

        # === LED power sets (nW for a 250 µm spot) ===
        self._POWER_SETS_NW = {
            "Current":                  {"Red": 1430.0, "Green": 168.0,  "Blue": 259.0,  "UV": 214.0},
            "test":                    {"Red": 3, "Green": 1,  "Blue": 0.97,  "UV": 0.41},
            "2025-04-30 – 2025-07-19": {"Red": 2856.0, "Green": 242.1,  "Blue": 813.0,  "UV": 887.0},
            "2024-11-06 – 2025-04-29": {"Red": 184.5,  "Green": 26.64,  "Blue": 40.69,  "UV": 31.15},
            "2024-06-05 – 2024-11-04": {"Red": 756.0,  "Green": 112.0,  "Blue": 410.0,  "UV": 370.0},
        }

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
                # Use the current trio order if possible; otherwise fall back to whatever is in the table
                leds3 = [L for L in leds if L in leds_all]
                if not leds3:
                    leds3 = leds_all

                code = _led_set_code(leds3)  # e.g. "RGB", "RGUV", "GBUV"

                for col in mods.columns:
                    # columns are like "LW_isolating", "MW_isolating", etc.
                    try:
                        vec = [float(mods.loc[L, col]) for L in leds3]
                    except Exception:
                        continue
                    ops_name = col.replace("_isolating", "")
                    # Label includes opsin + LED set, e.g. "LW (RGB)"
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
            # Surface Python exceptions instead of silent hard-exit
            QtWidgets.QMessageBox.critical(self, "Error in recompute()", str(e))
            raise
        finally:
            self._in_recompute = False

    def _save_matrices(self):
        # Prevent duplicate dialogs / re-entrant calls
        if self._saving:
            return
        self._saving = True
        try:
            if not self._last_out:
                QtWidgets.QMessageBox.warning(self, "Nothing to save", "Run a computation first.")
                return

            # === Ask user for destination folder ONCE ===
            dest_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Choose folder to save matrices", os.getcwd()
            )
            if not dest_dir:
                return

            out = self._last_out  # read-only use
            species = self.controls.current_species()
            species_tag = species.replace(" ", "_")

            # ND tag (robust if control returns None/empty)
            try:
                od_val = self.controls.nd_optical_density()
                nd_tag = f"ND{float(od_val):g}".replace(".", "p")
            except Exception:
                nd_tag = "ND0"

            # Timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            def save_df(df: pd.DataFrame, name: str, index=True):
                filename = f"{timestamp}_{species_tag}_{nd_tag}_{name}.csv"
                path = os.path.join(dest_dir, filename)
                df.copy().to_csv(path, float_format="%.6e", index=index)
                print(f"Saved {filename}")

            # --- Build labeled frames explicitly (no mutation) ---
            A_df    = out["ActivationMatrix"].copy()
            inv_df  = pd.DataFrame(out["InverseActivationMatrix"],
                                   index=out["opsins3"], columns=out["leds3"]).copy()
            P_df    = out["ProductMatrix"].copy()
            Psc_df  = out["ScaledProductMatrix"].copy()
            mods_df = out["Modulations"].copy()
            P_T_df  = P_df.T.copy()

            # --- Save ---
            save_df(A_df,   "ActivationMatrix")
            save_df(inv_df, "InverseActivationMatrix")
            save_df(P_df,   "ProductMatrix")
            save_df(Psc_df, "ScaledProductMatrix")
            save_df(mods_df,"Modulations")
            save_df(P_T_df, "ProductMatrix_T")

            # Optional: also save opsin-weighted R* if present
            iso_rates = out.get("iso_rates")
            if iso_rates and "Rstar_per_opsin" in iso_rates:
                rstar_series = pd.Series(iso_rates["Rstar_per_opsin"], name="Rstar_per_sec")
                save_df(rstar_series.to_frame(), "Rstar_per_opsin", index=True)

            QtWidgets.QMessageBox.information(self, "Saved", f"Matrices saved to:\n{dest_dir}")

        finally:
            self._saving = False

    def _recommend_led_ratios(self):
        """
        Compute recommended relative LED power multipliers (unitless, max=1)
        and the corresponding LED powers in nW for the currently selected
        calibration set and species/LED trio.
        """
        try:
            species = self.controls.current_species()
            leds = self.controls.selected_leds()
            power_key = self.controls.current_power_key()

            # Ensure we have an ActivationMatrix available
            if not self._last_out:
                pf = self._photon_flux
                wl_led = pf["Wavelength"].to_numpy(dtype=float)
                wl_master = np.arange(200.0, 701.0, 1.0)
                nomo_200_700 = generate_nomograms(species, wl_master, normalize=True)
                nomo = interpolate_nomograms(nomo_200_700, wl_led)
                out = isolation_pipeline(
                    photon_flux_df=pf,
                    nomograms_df=nomo,
                    species=species,
                    selected_leds_3=leds,
                    strategy=(
                        "match_across_isolations" if self.controls.match_across_isolations() else "per_isolation_max"),
                    use_exact_inverse=self.controls.use_exact_inverse(),
                )
            else:
                out = self._last_out

            A_full = out["ActivationMatrix"]
            ops3 = out["opsins3"]
            A0 = A_full.loc[leds, ops3]  # 3×3 base matrix

            # --- compute relative power ratios (max = 1.0) ---
            ratios = recommend_led_power_ratios(A0)

            # --- compute scaled absolute powers (nW) from current calibration set ---
            base_powers_nw = self._POWER_SETS_NW.get(power_key, self._POWER_SETS_NW["Current"])
            powers_scaled = {led: float(base_powers_nw.get(led, 0.0)) * float(ratios.get(led, 1.0))
                             for led in leds}

            # --- pretty print ---
            ratio_txt = ", ".join(f"{led}={val:.2f}" for led, val in ratios.items())
            power_txt = ", ".join(f"{led}={val:.1f} nW" for led, val in powers_scaled.items())

            print(f"[Recommended LED ratios] {ratio_txt}")
            print(f"[Recommended LED powers (nW, scaled from '{power_key}')] {power_txt}")

            # --- display in Photon-Flux tab ---
            if hasattr(self.plots, "set_ratio_summary"):
                # combine both into one line
                text = {
                    "ratios": ratios,
                    "powers": powers_scaled,
                    "label": power_key,
                }
                self.plots.set_ratio_summary(text)

            # --- status bar echo ---
            self.statusBar().showMessage(f"LED ratios {ratio_txt} | powers (nW): {power_txt}")

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "LED ratio recommendation failed", str(e))


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 760)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
