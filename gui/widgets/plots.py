import numpy as np
from PyQt6 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm

# ===== Color rules =====
LED_COLORS = {"Red": "red", "Green": "green", "Blue": "blue", "UV": "purple"}
LED_WAVELENGTHS = {"Red": 640, "Green": 550, "Blue": 465, "UV": 390}  # for ordering
OPSIN_COLORS = {
    "LW": "red",
    "MW": "green",
    "SW": "blue",
    "Rh": "orange",     # rho
    "Rho": "orange",    # just in case
    "UVS": "purple",    # mouse UV-sensitive
}


def _photon_flux_totals(photon_flux_df, leds, col_suffix="_PhotonFlux"):
    """
    Integrate per-λ photon flux (photons/s/µm²/nm) over wavelength (nm)
    to get total photons/s/µm² for each LED, plus a Sum.
    """
    if photon_flux_df is None or photon_flux_df.empty:
        return {led: float("nan") for led in leds} | {"Sum": float("nan")}
    wl = photon_flux_df["Wavelength"].to_numpy(float)  # nm
    totals = {}
    for led in leds:
        col = f"{led}{col_suffix}"
        if col in photon_flux_df.columns:
            phi = photon_flux_df[col].to_numpy(float)
            totals[led] = float(np.trapezoid(phi, wl))  # photons/s/µm²
    # Sum of finite entries
    totals["Sum"] = float(sum(v for v in totals.values() if np.isfinite(v)))
    return totals


def _ordered_leds(leds):
    """Return LEDs ordered longest → shortest wavelength."""
    return sorted(leds, key=lambda L: LED_WAVELENGTHS.get(L, 0), reverse=True)


def _led_set_code(leds_sorted):
    """
    Map an ordered LED set to a compact code for the vector label.
    Examples:
      ['Red','Green','Blue'] -> 'RGB'
      ['Red','Green','UV']   -> 'RGUV'
      ['Green','Blue','UV']  -> 'GBUV'
      otherwise join first letters ('R','G','B','U' for UV)
    """
    key = tuple(leds_sorted)
    if key == ("Red", "Green", "Blue"):
        return "RGB"
    if key == ("Red", "Green", "UV"):
        return "RGUV"
    if key == ("Green", "Blue", "UV"):
        return "GBUV"
    # Fallback: compose a code from initials
    def init(L):
        return "U" if L == "UV" else L[0]
    return "".join(init(L) for L in leds_sorted)


# ---------- Base matplotlib tab ----------
class _MplTab(QtWidgets.QWidget):
    def __init__(self, title=None):
        super().__init__()
        self.fig = Figure(constrained_layout=False)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        lay = QtWidgets.QVBoxLayout(self)
        if title:
            lbl = QtWidgets.QLabel(f"<b>{title}</b>")
            lay.addWidget(lbl)
        lay.addWidget(self.canvas)


# ---------- Photon Flux tab with totals strip ----------
class _PhotonFluxTab(QtWidgets.QWidget):
    """
    Plots per-λ photon flux and shows per-LED totals + Sum in photons/s/µm².
    """
    def __init__(self, title="Photon flux"):
        super().__init__()
        self.fig = Figure(constrained_layout=False)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        # stats strip (per-LED + Sum)
        self.stats = QtWidgets.QLabel("")
        self.stats.setTextFormat(QtCore.Qt.TextFormat.RichText)

        lay = QtWidgets.QVBoxLayout(self)
        if title:
            lay.addWidget(QtWidgets.QLabel(f"<b>{title}</b>"))
        lay.addWidget(self.canvas)
        lay.addWidget(self.stats)

    def update_plot(self, photon_flux_df, leds):
        ax = self.ax
        ax.clear()

        if photon_flux_df is None or photon_flux_df.empty or not leds:
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Photons / s / µm² / nm")
            ax.set_title("Photon flux")
            self.canvas.draw_idle()
            self._update_stats({}, [])
            return

        wl = photon_flux_df["Wavelength"].to_numpy(float)
        for led in _ordered_leds(leds):
            col = f"{led}_PhotonFlux"
            if col in photon_flux_df.columns:
                y = photon_flux_df[col].to_numpy(float)
                ax.plot(wl, y, label=led, color=LED_COLORS.get(led, None), linewidth=2)

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Photons / s / µm² / nm")
        ax.set_title("Photon flux (per-λ)")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

        totals = _photon_flux_totals(photon_flux_df, leds)
        self._update_stats(totals, leds)

    def _update_stats(self, totals_dict, leds):
        def fmt(x):
            if x is None or not np.isfinite(x):
                return "—"
            return f"{x:.3e}"
        bits = []
        for led in _ordered_leds(leds):
            color = LED_COLORS.get(led, "black")
            bits.append(f'<span style="color:{color}">{led}: {fmt(totals_dict.get(led))}</span>')
        bits.append(f"<b>Sum: {fmt(totals_dict.get('Sum'))}</b>")
        self.stats.setText(" &nbsp; | &nbsp; ".join(bits))


# ---------- Activation heatmap with fixed colorbar axis ----------
class _ActTab(QtWidgets.QWidget):
    """Activation heatmap with a fixed, dedicated colorbar axis."""
    def __init__(self, title=None):
        super().__init__()
        self.fig = Figure(constrained_layout=False)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
        self.ax  = self.fig.add_subplot(gs[0, 0])  # heatmap axis
        self.cax = self.fig.add_subplot(gs[0, 1])  # fixed colorbar axis
        self.canvas = FigureCanvas(self.fig)
        lay = QtWidgets.QVBoxLayout(self)
        if title:
            lbl = QtWidgets.QLabel(f"<b>{title}</b>")
            lay.addWidget(lbl)
        lay.addWidget(self.canvas)
        self._last_norm = None
        self._last_cmap = None


# ---------- Isolation tab with modulation vector readout ----------
class _IsolationTab(QtWidgets.QWidget):
    """
    Shows per-LED modulation bars for the selected opsin, with an RGB/RGUV/GBUV
    vector readout like: RGB: 1.00, -0.50, 0.22
    """
    def __init__(self):
        super().__init__()
        self.combo = QtWidgets.QComboBox()
        self.fig = Figure(constrained_layout=False)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        # vector readout (right side of the header row)
        self.vec_label = QtWidgets.QLabel("")
        self.vec_label.setTextFormat(QtCore.Qt.TextFormat.RichText)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Show isolation for:"))
        header.addWidget(self.combo)
        header.addStretch(1)
        header.addWidget(self.vec_label)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(header)
        lay.addWidget(self.canvas)

        self.combo.currentIndexChanged.connect(self._redraw)

        self._mods = None          # DataFrame: index = LEDs, columns = '<opsin>_isolating'
        self._leds_in_use = None   # list like ['Red','Green','Blue'] (for ordering & code)

    def set_modulations(self, mods_df, leds_in_use):
        """mods_df: DataFrame with rows=LEDs and columns '<opsin>_isolating'."""
        self._mods = mods_df
        self._leds_in_use = list(leds_in_use) if leds_in_use else None

        self.combo.blockSignals(True)
        self.combo.clear()
        if mods_df is not None and not mods_df.empty:
            self.combo.addItems(list(mods_df.columns))  # '<opsin>_isolating'
        self.combo.blockSignals(False)
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        if self._mods is None or self._mods.empty or self.combo.count() == 0:
            self.ax.text(0.5, 0.5, "No isolation vectors", ha="center", va="center")
            self.vec_label.setText("")
            self.canvas.draw_idle()
            return

        col = self.combo.currentText()     # e.g., "LW_isolating"
        # Use LEDs present in the mods table, restricted/ordered by current config
        leds_all = list(self._mods.index)
        leds = [L for L in (_ordered_leds(self._leds_in_use or leds_all)) if L in leds_all]
        if not leds:
            leds = _ordered_leds(leds_all)

        vals = self._mods.loc[leds, col].to_numpy(float)
        bar_colors = [LED_COLORS.get(L, "gray") for L in leds]

        # Bar plot
        self.ax.bar(leds, vals, color=bar_colors)
        self.ax.set_title(f"LED modulation for {col}")
        self.ax.axhline(0, color="black", linewidth=0.8)
        self.ax.set_ylabel("Relative LED modulation")
        self.canvas.draw_idle()

        # Vector readout like "RGB: 1.00, -0.50, 0.22"
        code = _led_set_code(leds)
        vec_txt = ", ".join(f"{v:+.2f}".replace("+", "") for v in vals)  # drop '+' for positives
        # color the initials inline
        initials = []
        for L in leds:
            initial = "U" if L == "UV" else L[0]
            color = LED_COLORS.get(L, "black")
            initials.append(f'<span style="color:{color}">{initial}</span>')
        label_html = f"<b>{''.join(initials)}:</b> {vec_txt}"
        # Use standard code (RGB/RGUV/GBUV) if it matches a known set
        if code in ("RGB", "RGUV", "GBUV"):
            label_html = f"<b>{code}:</b> {vec_txt}"
        self.vec_label.setText(label_html)


# ---------- Container ----------
class Plots(QtWidgets.QTabWidget):
    def __init__(self):
        super().__init__()
        # Tabs
        self.tab_flux  = _PhotonFluxTab("Photon flux (selected LEDs)")
        self.tab_nomo  = _MplTab("Opsin nomograms")
        self.tab_act   = _ActTab("Activation matrix (LED × Opsin)")
        self.tab_iso   = _IsolationTab()

        self.addTab(self.tab_flux, "Photon Flux")
        self.addTab(self.tab_nomo, "Nomograms")
        self.addTab(self.tab_act,  "Activation")
        self.addTab(self.tab_iso,  "Isolation")

    # ---- Public entry point from controller ----
    def update_all(
        self,
        photon_flux_df,
        nomograms_df,
        activation_matrix,
        modulations_df,
        selected_leds,
        species: str,
    ):
        self._update_flux(photon_flux_df, selected_leds)
        self._update_nomograms(nomograms_df, species)
        self._update_activation(activation_matrix)
        self.tab_iso.set_modulations(modulations_df, selected_leds)

    def update_photon_flux(self, photon_flux_df, leds):
        # Exposed for callers that want to refresh flux alone
        self.tab_flux.update_plot(photon_flux_df, leds)

    # ---- Tab renderers ----
    def _update_flux(self, pf_df, selected_leds):
        self.tab_flux.update_plot(pf_df, selected_leds)

    def _update_nomograms(self, nomo_df, species):
        ax = self.tab_nomo.ax
        ax.clear()
        if nomo_df is None or nomo_df.empty:
            ax.text(0.5, 0.5, "No nomograms", ha="center", va="center")
            self.tab_nomo.canvas.draw_idle()
            return

        wl = nomo_df["Wavelength"].to_numpy(float)
        for col in nomo_df.columns:
            if col == "Wavelength":
                continue
            ax.plot(
                wl,
                nomo_df[col].to_numpy(float),
                label=col,
                color=OPSIN_COLORS.get(col, None),
                linewidth=2,
            )
        ax.set_title(f"Nomograms – {species}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized sensitivity (area = 1)")
        ax.legend(frameon=False, ncol=2)
        ax.grid(True, alpha=0.3)
        self.tab_nomo.canvas.draw_idle()

    def _update_activation(self, A):
        self.tab_act.ax.clear()
        self.tab_act.cax.clear()

        if A is None or A.empty:
            self.tab_act.ax.text(0.5, 0.5, "No data", ha="center", va="center")
            self.tab_act.canvas.draw_idle()
            return

        # Heatmap
        im = self.tab_act.ax.imshow(A.values, aspect="auto")

        # Axes labels
        self.tab_act.ax.set_xticks(range(len(A.columns)))
        self.tab_act.ax.set_xticklabels(A.columns, rotation=45, ha="right")
        self.tab_act.ax.set_yticks(range(len(A.index)))
        self.tab_act.ax.set_yticklabels(A.index)
        self.tab_act.ax.set_title("Activation (LED × Opsin)")

        # ---- Annotate each cell with scientific notation ----
        data = A.to_numpy(dtype=float)
        # fallback linear normalization if im.norm not present
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        denom = (vmax - vmin) if np.isfinite(vmax - vmin) and (vmax - vmin) > 0 else 1.0

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    s = "nan"
                    bg_level = 0.5
                else:
                    s = f"{val:.2e}"
                    # map value to 0..1 for contrast check
                    if getattr(im, "norm", None) is not None:
                        try:
                            bg_level = im.norm(val)
                        except Exception:
                            bg_level = (val - vmin) / denom
                    else:
                        bg_level = (val - vmin) / denom
                color = "white" if bg_level > 0.5 else "black"
                self.tab_act.ax.text(j, i, s, ha="center", va="center", fontsize=8, color=color)

        # Colorbar in the dedicated axis (stable layout)
        cmappable = cm.ScalarMappable(norm=im.norm, cmap=im.cmap)
        ColorbarBase(self.tab_act.cax, cmap=im.cmap, norm=im.norm, orientation="vertical")
        self.tab_act.cax.set_ylabel("Activation (∫ φ(λ)·S(λ) dλ)")

        self.tab_act.canvas.draw_idle()

