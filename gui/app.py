from PyQt6 import QtWidgets, QtCore
from opsinlab.io_led import process_led_data
from opsinlab.nomograms import generate_nomograms, available_species
from opsinlab.activation import isolation_pipeline
from gui.widgets.controls import Controls
from gui.widgets.plots import Plots

import os
from datetime import datetime
import pandas as pd


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpsinActivation")

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

        # last outputs cache for saving
        self._last_out = None

        # wire events
        self.controls.paramsChanged.connect(self.recompute)
        self.controls.saveMatricesClicked.connect(self._save_matrices)  # <-- NEW

        # initial compute
        QtCore.QTimer.singleShot(0, self.recompute)

    def recompute(self):
        species = self.controls.current_species()
        leds = self.controls.selected_leds()
        wl = self._photon_flux["Wavelength"].values
        nomo = generate_nomograms(species, wl, normalize=True)

        strategy = "match_across_isolations" if self.controls.match_across_isolations() else "per_isolation_max"
        use_inv = self.controls.use_exact_inverse()

        out = isolation_pipeline(
            photon_flux_df=self._photon_flux,
            nomograms_df=nomo,
            species=species,
            selected_leds_3=leds,
            strategy=strategy,
            use_exact_inverse=use_inv,
        )

        # cache & enable save button
        self._last_out = out
        self.controls.save_btn.setEnabled(True)

        # update plots
        A = out["ActivationMatrix"]
        mods = out["Modulations"]
        self.plots.update_all(
            photon_flux_df=self._photon_flux,
            nomograms_df=nomo,
            activation_matrix=A,
            modulations_df=mods,
            selected_leds=leds,
            species=species,
        )

    # --- NEW: Save matrices only on demand ---
    def _save_matrices(self):
        if not self._last_out:
            QtWidgets.QMessageBox.warning(self, "Nothing to save", "Run a computation first.")
            return

        # choose folder
        dest_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose folder to save matrices", os.getcwd()
        )
        if not dest_dir:
            return

        out = self._last_out
        species = self.controls.current_species()
        species_tag = species.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def save_df(df: pd.DataFrame, name: str):
            path = os.path.join(dest_dir, f"{timestamp}_{species_tag}_{name}.csv")
            df.to_csv(path, float_format="%.6e", index=True)

        # ActivationMatrix (LED×Opsin)
        save_df(out["ActivationMatrix"], "ActivationMatrix")

        # InverseActivationMatrix (3×3, numpy -> DataFrame with labels)
        inv = pd.DataFrame(out["InverseActivationMatrix"],
                           index=out["opsins3"], columns=out["leds3"])
        save_df(inv, "InverseActivationMatrix")

        # ProductMatrix (LED×<opsin>_ISO)
        save_df(out["ProductMatrix"], "ProductMatrix")

        # ScaledProductMatrix (LED×<opsin>_ISO, scaled)
        save_df(out["ScaledProductMatrix"], "ScaledProductMatrix")

        # Modulations (LED×<opsin>_isolating)
        save_df(out["Modulations"], "Modulations")

        # Optional: also save transposed Product (opsin as rows) for spreadsheet parity
        save_df(out["ProductMatrix"].T, "ProductMatrix_T")

        QtWidgets.QMessageBox.information(self, "Saved", f"Matrices saved to:\n{dest_dir}")


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 760)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
