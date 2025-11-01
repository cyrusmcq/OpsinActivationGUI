from typing import List
from PyQt6 import QtWidgets, QtCore

_LED_CONFIGS = {
    "RGB":  ["Red", "Green", "Blue"],
    "RGUV": ["Red", "Green", "UV"],
    "GBUV": ["Green", "Blue", "UV"],
}

class Controls(QtWidgets.QWidget):
    # Public signals the MainWindow listens to
    paramsChanged = QtCore.pyqtSignal()
    saveMatricesClicked = QtCore.pyqtSignal()

    def __init__(self, species_list: List[str], led_configs: List[str]):
        super().__init__()

        v = QtWidgets.QVBoxLayout(self)

        # --- Species ---
        v.addWidget(QtWidgets.QLabel("Species"))
        self.species_combo = QtWidgets.QComboBox()
        self.species_combo.addItems(species_list)
        v.addWidget(self.species_combo)

        # --- LED configuration (exactly 3 at a time) ---
        v.addWidget(QtWidgets.QLabel("LED configuration (3 LEDs)"))
        self.led_combo = QtWidgets.QComboBox()
        self.led_combo.addItems(led_configs)
        v.addWidget(self.led_combo)

        # --- Neutral Density (optical density) ---
        v.addWidget(QtWidgets.QLabel("Neutral Density (OD)"))
        self.nd_combo = QtWidgets.QComboBox()
        self.nd_combo.addItems(["0", "0.25", "1", "2", "3"])
        self.nd_combo.setToolTip("Optical density; transmission = 10^(-OD)")
        v.addWidget(self.nd_combo)

        # --- PMT lever (70/30 mirror) ---
        self.pmt_chk = QtWidgets.QCheckBox("PMT lever in (70/30 → 30% to sample)")
        self.pmt_chk.setChecked(False)  # default: unchecked = full flux
        v.addWidget(self.pmt_chk)

        # --- Options ---
        self.equalize_chk = QtWidgets.QCheckBox("Match modulation across isolations")
        self.equalize_chk.setChecked(True)
        v.addWidget(self.equalize_chk)

        self.exactinv_chk = QtWidgets.QCheckBox("Use exact inverse (3×3)")
        self.exactinv_chk.setChecked(True)
        v.addWidget(self.exactinv_chk)

        # --- Save button ---
        self.save_btn = QtWidgets.QPushButton("Save matrices to CSV…")
        self.save_btn.setEnabled(False)  # enabled after first compute
        v.addWidget(self.save_btn)

        v.addStretch(1)

        # --- Events (hook all widgets to one emitter) ---
        self.species_combo.currentIndexChanged.connect(self._emit_params_changed)
        self.led_combo.currentIndexChanged.connect(self._emit_params_changed)
        self.nd_combo.currentIndexChanged.connect(self._emit_params_changed)
        self.equalize_chk.stateChanged.connect(self._emit_params_changed)
        self.exactinv_chk.stateChanged.connect(self._emit_params_changed)
        self.pmt_chk.stateChanged.connect(self._emit_params_changed)

        # Save button emits a separate signal
        self.save_btn.clicked.connect(lambda *_: self.saveMatricesClicked.emit())

    # ---- helpers / getters ----
    def _emit_params_changed(self, *_):
        """Unify all control changes into a single signal."""
        self.paramsChanged.emit()

    def current_species(self) -> str:
        return self.species_combo.currentText()

    def selected_leds(self) -> List[str]:
        key = self.led_combo.currentText()
        return list(_LED_CONFIGS.get(key, []))

    def nd_optical_density(self) -> float:
        """Return selected neutral density (OD)."""
        try:
            return float(self.nd_combo.currentText())
        except Exception:
            return 0.0

    def pmt_lever_enabled(self) -> bool:
        return self.pmt_chk.isChecked()

    def match_across_isolations(self) -> bool:
        return self.equalize_chk.isChecked()

    def use_exact_inverse(self) -> bool:
        return self.exactinv_chk.isChecked()
