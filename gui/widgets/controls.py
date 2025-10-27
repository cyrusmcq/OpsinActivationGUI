from typing import List
from PyQt6 import QtWidgets, QtCore

_LED_CONFIGS = {
    "RGB":  ["Red", "Green", "Blue"],
    "RGUV": ["Red", "Green", "UV"],
    "GBUV": ["Green", "Blue", "UV"],
}

class Controls(QtWidgets.QWidget):
    paramsChanged = QtCore.pyqtSignal()
    saveMatricesClicked = QtCore.pyqtSignal()   # <-- NEW

    def __init__(self, species_list: List[str], led_configs: List[str]):
        super().__init__()

        v = QtWidgets.QVBoxLayout(self)

        # Species
        v.addWidget(QtWidgets.QLabel("Species"))
        self.species_combo = QtWidgets.QComboBox()
        self.species_combo.addItems(species_list)
        v.addWidget(self.species_combo)

        # LED configuration (exactly 3 at a time)
        v.addWidget(QtWidgets.QLabel("LED configuration (3 LEDs)"))
        self.led_combo = QtWidgets.QComboBox()
        self.led_combo.addItems(led_configs)
        v.addWidget(self.led_combo)

        # Options
        self.equalize_chk = QtWidgets.QCheckBox("Match modulation across isolations")
        self.equalize_chk.setChecked(True)
        v.addWidget(self.equalize_chk)

        self.exactinv_chk = QtWidgets.QCheckBox("Use exact inverse (3×3)")
        self.exactinv_chk.setChecked(True)
        v.addWidget(self.exactinv_chk)

        # --- NEW: Save button ---
        self.save_btn = QtWidgets.QPushButton("Save matrices to CSV…")
        self.save_btn.setEnabled(False)  # enabled after first compute
        v.addWidget(self.save_btn)

        v.addStretch(1)

        # Events
        self.species_combo.currentIndexChanged.connect(self.paramsChanged)
        self.led_combo.currentIndexChanged.connect(self.paramsChanged)
        self.equalize_chk.stateChanged.connect(self.paramsChanged)
        self.exactinv_chk.stateChanged.connect(self.paramsChanged)
        self.save_btn.clicked.connect(self.saveMatricesClicked)  # <-- NEW

    # --- getters used by app.py ---
    def current_species(self) -> str:
        return self.species_combo.currentText()

    def selected_leds(self) -> List[str]:
        key = self.led_combo.currentText()
        return list(_LED_CONFIGS.get(key, []))

    def match_across_isolations(self) -> bool:
        return self.equalize_chk.isChecked()

    def use_exact_inverse(self) -> bool:
        return self.exactinv_chk.isChecked()
