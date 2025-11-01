from PyQt6 import QtWidgets, QtCore
from opsinlab.io_led import process_led_data
from opsinlab.nomograms import generate_nomograms, available_species
from opsinlab.activation import isolation_pipeline
from gui.widgets.controls import Controls
from gui.widgets.plots import Plots
from opsinlab.activation import opsin_weighted_isomerizations

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

        # cache photon flux (all LEDs available)
        self._photon_flux = process_led_data()
        self._photon_flux_base = self._photon_flux.copy()  # pristine (no ND)

        # last outputs cache for saving
        self._last_out = None

        # re-entrancy guard to prevent signal loops
        self._in_recompute = False

        # wire events
        self.controls.paramsChanged.connect(self.recompute)
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
            # scale both styles of columns if present
            for col in pf.columns:
                if col.endswith("_PhotonFlux") or col in ("Red", "Green", "Blue", "UV"):
                    pf[col] = pf[col] * transmission
            self._photon_flux = pf

            # Nomograms on same wavelength grid
            wl = pf["Wavelength"].values
            nomo = generate_nomograms(species, wl, normalize=True)

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

            # cache & enable save button
            self._last_out = out | {"iso_rates": iso_rates}
            self.controls.save_btn.setEnabled(True)

            # update plots once
            A = out["ActivationMatrix"]
            mods = out["Modulations"]
            self.plots.update_all(
                photon_flux_df=pf,
                nomograms_df=nomo,
                activation_matrix=A,
                modulations_df=mods,
                selected_leds=leds,
                species=species,
            )
            if hasattr(self.plots.tab_flux, "set_isom_summary"):
                self.plots.tab_flux.set_isom_summary(iso_rates)

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


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 760)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
