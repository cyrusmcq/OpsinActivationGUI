import numpy as np
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
import os
import pandas as pd


# ====== Data structures ======

@dataclass
class LedCalibration:
    """Calibration data for a single LED channel."""
    name: str
    currents_pct: np.ndarray  # calibration currents (%)
    powers_uW: np.ndarray     # calibration powers (µW)
    a: float = None           # fitted prefactor
    gamma: float = None       # fitted exponent


@dataclass
class BitmaskChannelData:
    """Experiment-mode bitmask data for one LED channel."""
    name: str
    bitmasks: np.ndarray      # e.g. [1,2,4,8,16,32,64,128]
    currents_pct: np.ndarray  # per-bit currents in experiment mode (%)
    powers_uW: np.ndarray     # per-bit powers in experiment mode (µW)


# ====== Bitmask targets (anchored at MSB) ======

def compute_bit_targets(
    powers_uW: np.ndarray,
    anchor_index: int = -1,
) -> np.ndarray:
    """
    Given measured bit-plane powers in experiment mode, compute ideal
    target powers for perfect binary doubling, anchored at MSB.

    bits assumed ordered from LSB -> MSB.
    anchor_index=-1 means last element is the MSB.
    """
    P_meas = np.array(powers_uW, dtype=float)
    B = P_meas.size

    if anchor_index < 0:
        anchor_index = B + anchor_index

    P_anchor = P_meas[anchor_index]
    exponents = np.arange(B) - anchor_index
    P_target = P_anchor * (2.0 ** exponents)
    return P_target


# ====== Current correction per bit ======

def interpolate_current_from_power(
    I_cal_pct: np.ndarray,
    P_cal_uW: np.ndarray,
    P_target_uW: float,
) -> float:
    """
    Given calibration arrays I_cal_pct (monotonic) and P_cal_uW (monotonic),
    find current I_new such that P_cal(I_new) ~= P_target_uW
    by 1D interpolation.

    Assumes P_cal_uW is monotonically increasing with current.
    """
    I_cal = np.array(I_cal_pct, dtype=float)
    P_cal = np.array(P_cal_uW, dtype=float)

    # Clamp target within calibration range to avoid extrapolation craziness
    P_min, P_max = P_cal.min(), P_cal.max()
    P_t = np.clip(P_target_uW, P_min, P_max)

    # np.interp: x -> y, here x = P_cal, y = I_cal
    I_new = np.interp(P_t, P_cal, I_cal)
    return float(I_new)


def correct_currents_for_bitmask(
    I_cal_pct: np.ndarray,
    P_cal_uW: np.ndarray,
    I_old_pct: np.ndarray,
    P_meas_uW: np.ndarray,
    P_target_uW: np.ndarray,
    anchor_index: int = -1,
) -> np.ndarray:
    """
    Interpolation-based correction using the calibration curve directly.

    For each bit:
      P_cal_old = P_cal(I_old)
      P_cal_target = P_cal_old * (P_target / P_meas)
      I_new = I_cal^-1(P_cal_target) via interpolation

    Anchor bit (MSB) is kept fixed.
    """
    I_old = np.array(I_old_pct, dtype=float)
    P_meas = np.array(P_meas_uW, dtype=float)
    P_target = np.array(P_target_uW, dtype=float)

    B = I_old.size
    if anchor_index < 0:
        anchor_index = B + anchor_index

    I_new = np.empty_like(I_old)

    # Pre-calc calibration arrays
    I_cal = np.array(I_cal_pct, dtype=float)
    P_cal = np.array(P_cal_uW, dtype=float)

    for b in range(B):
        if b == anchor_index:
            # keep MSB fixed
            I_new[b] = I_old[b]
            continue

        if P_meas[b] <= 0:
            # no reliable measurement: don't change this bit
            I_new[b] = I_old[b]
            continue

        # Calibration power at the old current
        P_cal_old = np.interp(I_old[b], I_cal, P_cal)

        # What calibration power would correspond to the desired ratio?
        scale = P_target[b] / P_meas[b]
        P_cal_target = P_cal_old * scale

        # Invert P_cal via interpolation
        I_new[b] = interpolate_current_from_power(I_cal, P_cal, P_cal_target)

    return I_new


def linearize_led_bitmasks(
    cal: LedCalibration,
    bit_data: BitmaskChannelData,
    anchor_index: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolation-based linearization for one LED channel.

    Returns:
        P_target_uW, I_new_pct, ratio = P_target/P_meas
    """
    I_cal = cal.currents_pct
    P_cal = cal.powers_uW

    # Targets for perfect doubling, anchored at MSB
    P_target = compute_bit_targets(bit_data.powers_uW, anchor_index=anchor_index)

    I_new = correct_currents_for_bitmask(
        I_cal_pct=I_cal,
        P_cal_uW=P_cal,
        I_old_pct=bit_data.currents_pct,
        P_meas_uW=bit_data.powers_uW,
        P_target_uW=P_target,
        anchor_index=anchor_index,
    )

    ratio = P_target / np.array(bit_data.powers_uW, dtype=float)
    return P_target, I_new, ratio



# ====== Printing helpers: RGB and GBUV formats ======

def print_rgb_table(I_red: np.ndarray, I_green: np.ndarray, I_blue: np.ndarray) -> None:
    """
    Print currents in 'RGB' format:
      LED 4 = Red, LED 3 = Green, LED 2 = Blue
      Starting from MSB down to LSB.
    """
    print("LED #\tLED PWM (%)\tLED current (%)")
    B = len(I_red)
    for idx in range(B - 1, -1, -1):  # MSB to LSB
        print(f"4\t100\t{I_red[idx]:.3f}")
        print(f"3\t100\t{I_green[idx]:.3f}")
        print(f"2\t100\t{I_blue[idx]:.3f}")


def print_gbuv_table(I_green: np.ndarray, I_blue: np.ndarray, I_uv: np.ndarray) -> None:
    """
    Print currents in 'GBUV' format:
      LED 3 = Green, LED 2 = Blue, LED 1 = UV
      Starting from MSB down to LSB.
    """
    print("LED #\tLED PWM (%)\tLED current (%)")
    B = len(I_green)
    for idx in range(B - 1, -1, -1):  # MSB to LSB
        print(f"3\t100\t{I_green[idx]:.3f}")
        print(f"2\t100\t{I_blue[idx]:.3f}")
        print(f"1\t100\t{I_uv[idx]:.3f}")

def plot_calibration_curve(cal: LedCalibration):
    """
    Plot only the measured calibration data (and an interpolated curve),
    no power-law fitting.
    """
    I = np.array(cal.currents_pct, dtype=float)
    P = np.array(cal.powers_uW, dtype=float)

    # Sort in case
    order = np.argsort(I)
    I = I[order]
    P = P[order]

    # Simple dense interpolation for a smooth curve
    I_dense = np.linspace(I.min(), I.max(), 200)
    P_dense = np.interp(I_dense, I, P)

    plt.figure()
    plt.loglog(I, P, "o", label="Calibration data")
    plt.loglog(I_dense, P_dense, "-", label="Interpolated calibration")
    plt.xlabel("Current (%)")
    plt.ylabel("Power (µW)")
    plt.title(f"Calibration curve: {cal.name}")
    plt.legend()
    plt.grid(True, which="both")


def plot_bitmask_powers(bit_data: BitmaskChannelData,
                        P_target: np.ndarray,
                        title_suffix: str = ""):
    """
    Plot measured vs target bit-plane powers in experiment mode.
    """
    bits = np.arange(len(bit_data.bitmasks))  # 0..7 (LSB..MSB)
    P_meas = np.array(bit_data.powers_uW, dtype=float)

    plt.figure()
    plt.plot(bits, P_meas, "o-", label="Measured experiment power")
    plt.plot(bits, P_target, "s--", label="Target power (binary ladder)")
    plt.xticks(bits, bit_data.bitmasks)
    plt.xlabel("Bit mask")
    plt.ylabel("Power (µW)")
    plt.title(f"Bit-plane powers: {bit_data.name} {title_suffix}")
    plt.legend()
    plt.grid(True)


def plot_bitmask_currents(bit_data: BitmaskChannelData,
                          I_new_pct: np.ndarray,
                          title_suffix: str = ""):
    """
    Plot original vs corrected currents per bit.
    """
    bits = np.arange(len(bit_data.bitmasks))  # 0..7
    I_old = np.array(bit_data.currents_pct, dtype=float)
    I_new = np.array(I_new_pct, dtype=float)

    plt.figure()
    plt.plot(bits, I_old, "o-", label="Original current (%)")
    plt.plot(bits, I_new, "s--", label="Corrected current (%)")
    plt.xticks(bits, bit_data.bitmasks)
    plt.xlabel("Bit mask")
    plt.ylabel("LED current (%)")
    plt.title(f"Bit-plane currents: {bit_data.name} {title_suffix}")
    plt.legend()
    plt.grid(True)


def compute_linearized_defaults():
    """
    Reproduce the hard-coded example from __main__ but return the
    linearized bit currents for R, G, B, UV instead of just plotting/printing.

    Returns
    -------
    I_new_red, I_new_green, I_new_blue, I_new_uv : np.ndarray
        Each is length 8, ordered LSB→MSB (bitmasks 1,2,...,128).
    """
    # Calibration currents (%)
    currents = np.array(
        [2.8, 3, 3.25, 3.5, 4, 5, 6, 7, 8, 9,
         10, 12, 15, 20, 30, 40, 50, 60, 80, 100],
        dtype=float
    )

    # Calibration powers (µW) for each channel
    red_cal_p = np.array(
        [0.00017, 0.00039, 0.00070, 0.00099, 0.00157,
         0.00279, 0.00404, 0.00530, 0.00655, 0.00778,
         0.00905, 0.01175, 0.01560, 0.02177, 0.03429,
         0.04642, 0.05719, 0.06700, 0.08487, 0.10100]
    )
    green_cal_p = np.array(
        [0.00015, 0.00032, 0.00057, 0.00081, 0.00128,
         0.00221, 0.00315, 0.00408, 0.00499, 0.00588,
         0.00676, 0.00849, 0.01111, 0.01498, 0.02263,
         0.02922, 0.03562, 0.04138, 0.05238, 0.06180]
    )
    blue_cal_p = np.array(
        [0.00021, 0.00047, 0.00084, 0.00119, 0.00187,
         0.00326, 0.00469, 0.00599, 0.00739, 0.00863,
         0.00997, 0.01240, 0.01608, 0.02175, 0.03201,
         0.04125, 0.04986, 0.05727, 0.07063, 0.08220]
    )
    uv_cal_p = np.array(
        [0.00069, 0.00161, 0.00290, 0.00410, 0.00650,
         0.01153, 0.01673, 0.02177, 0.02725, 0.03228,
         0.03747, 0.04755, 0.06267, 0.08785, 0.13690,
         0.18476, 0.23084, 0.27588, 0.36601, 0.44750]
    )

    red_cal = LedCalibration("Red", currents, red_cal_p)
    green_cal = LedCalibration("Green", currents, green_cal_p)
    blue_cal = LedCalibration("Blue", currents, blue_cal_p)
    uv_cal = LedCalibration("UV", currents, uv_cal_p)

    # Bitmask definitions (LSB -> MSB)
    bitmasks = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=int)

    # Experiment-mode bit currents (%), in ascending bit order (1..128)
    red_curr = np.array([3.329, 4.004, 5.288, 7.807, 12.671, 22.763, 43.741, 99.876])
    green_curr = np.array([3.164, 3.664, 4.695, 6.757, 11.101, 20.587, 42.541, 99.788])
    blue_curr = np.array([3.114, 3.571, 4.501, 6.344, 10.251, 18.941, 39.835, 99.983])
    uv_curr = np.array([3.374, 4.097, 5.472, 8.142, 13.662, 24.894, 48.451, 99.975])

    # Experiment-mode powers per bit (µW), ascending bit order
    red_p_exp = np.array([0.40, 0.70, 1.40, 2.80, 5.50, 11.10, 22.50, 52.00])
    green_p_exp = np.array([0.20, 0.30, 0.50, 0.80, 1.50, 3.10, 6.50, 13.70])
    blue_p_exp = np.array([0.10, 0.10, 0.40, 0.70, 1.40, 2.90, 5.60, 11.30])
    uv_p_exp = np.array([0.10, 0.10, 0.20, 0.30, 0.60, 1.20, 2.30, 4.50])

    red_bits   = BitmaskChannelData("Red",   bitmasks, red_curr,   red_p_exp)
    green_bits = BitmaskChannelData("Green", bitmasks, green_curr, green_p_exp)
    blue_bits  = BitmaskChannelData("Blue",  bitmasks, blue_curr,  blue_p_exp)
    uv_bits    = BitmaskChannelData("UV",    bitmasks, uv_curr,    uv_p_exp)

    # Linearize each channel (anchor MSB = last index)
    _, I_new_red, _   = linearize_led_bitmasks(red_cal,   red_bits,   anchor_index=-1)
    _, I_new_green, _ = linearize_led_bitmasks(green_cal, green_bits, anchor_index=-1)
    _, I_new_blue, _  = linearize_led_bitmasks(blue_cal,  blue_bits,  anchor_index=-1)
    _, I_new_uv, _    = linearize_led_bitmasks(uv_cal,    uv_bits,    anchor_index=-1)

    return I_new_red, I_new_green, I_new_blue, I_new_uv

def export_linearized_to_csv(
    assets_dir: str,
    rgb_template: str = "RGB.csv",
    gbuv_template: str = "GBUV.csv",
    rgb_out: str = "RGB_suggested.csv",
    gbuv_out: str = "GBUV_suggested.csv",
) -> tuple[str, str]:
    """
    Use the default calibration + experiment data to compute linearized
    bit-plane currents, then write new RGB/GBUV CSVs in the same layout
    as the templates located in `assets_dir`.

    Parameters
    ----------
    assets_dir : str
        Folder containing RGB.csv and GBUV.csv (original template files).
    rgb_template, gbuv_template : str
        Template filenames inside assets_dir.
    rgb_out, gbuv_out : str
        Output filenames to write inside assets_dir.

    Returns
    -------
    (rgb_path_out, gbuv_path_out) : tuple[str, str]
        Full paths to the written CSV files.
    """
    assets_dir = os.path.abspath(assets_dir)
    os.makedirs(assets_dir, exist_ok=True)

    rgb_path_in  = os.path.join(assets_dir, rgb_template)
    gbuv_path_in = os.path.join(assets_dir, gbuv_template)

    if not os.path.exists(rgb_path_in):
        raise FileNotFoundError(f"RGB template not found: {rgb_path_in}")
    if not os.path.exists(gbuv_path_in):
        raise FileNotFoundError(f"GBUV template not found: {gbuv_path_in}")

    rgb_df  = pd.read_csv(rgb_path_in)
    gbuv_df = pd.read_csv(gbuv_path_in)

    I_new_red, I_new_green, I_new_blue, I_new_uv = compute_linearized_defaults()

    # Both CSVs are laid out as 8 groups of 3 rows (MSB → LSB).
    # Our I_new_* arrays are LSB → MSB, so we reverse the mapping.
    B = len(I_new_red)
    if B != 8:
        raise ValueError(f"Expected 8 bits; got {B}")

    rgb_df = rgb_df.copy()
    gbuv_df = gbuv_df.copy()

    # --- RGB: LED 4=Red, LED 3=Green, LED 2=Blue ---
    for b in range(B):  # b = 0..7 (LSB..MSB)
        row_base = (B - 1 - b) * 3  # MSB group starts at row 0
        rgb_df.loc[row_base,     "LED current (%)"] = I_new_red[b]
        rgb_df.loc[row_base + 1, "LED current (%)"] = I_new_green[b]
        rgb_df.loc[row_base + 2, "LED current (%)"] = I_new_blue[b]

    # --- GBUV: LED 3=Green, LED 2=Blue, LED 1=UV ---
    for b in range(B):  # b = 0..7
        row_base = (B - 1 - b) * 3
        gbuv_df.loc[row_base,     "LED current (%)"] = I_new_green[b]
        gbuv_df.loc[row_base + 1, "LED current (%)"] = I_new_blue[b]
        gbuv_df.loc[row_base + 2, "LED current (%)"] = I_new_uv[b]

    rgb_path_out  = os.path.join(assets_dir, rgb_out)
    gbuv_path_out = os.path.join(assets_dir, gbuv_out)

    rgb_df.to_csv(rgb_path_out, index=False)
    gbuv_df.to_csv(gbuv_path_out, index=False)

    print(f"[LED calibration] Wrote {rgb_path_out}")
    print(f"[LED calibration] Wrote {gbuv_path_out}")

    return rgb_path_out, gbuv_path_out

# ====== Main: define your data and run everything ======

if __name__ == "__main__":
    # Calibration currents (%)
    currents = np.array(
        [2.8, 3, 3.25, 3.5, 4, 5, 6, 7, 8, 9,
         10, 12, 15, 20, 30, 40, 50, 60, 80, 100],
        dtype=float
    )

    # Calibration powers (µW) for each channel
    red_cal_p = np.array(
        [0.00017, 0.00039, 0.00070, 0.00099, 0.00157,
         0.00279, 0.00404, 0.00530, 0.00655, 0.00778,
         0.00905, 0.01175, 0.01560, 0.02177, 0.03429,
         0.04642, 0.05719, 0.06700, 0.08487, 0.10100]
    )
    green_cal_p = np.array(
        [0.00015, 0.00032, 0.00057, 0.00081, 0.00128,
         0.00221, 0.00315, 0.00408, 0.00499, 0.00588,
         0.00676, 0.00849, 0.01111, 0.01498, 0.02263,
         0.02922, 0.03562, 0.04138, 0.05238, 0.06180]
    )
    blue_cal_p = np.array(
        [0.00021, 0.00047, 0.00084, 0.00119, 0.00187,
         0.00326, 0.00469, 0.00599, 0.00739, 0.00863,
         0.00997, 0.01240, 0.01608, 0.02175, 0.03201,
         0.04125, 0.04986, 0.05727, 0.07063, 0.08220]
    )
    uv_cal_p = np.array(
        [0.00069, 0.00161, 0.00290, 0.00410, 0.00650,
         0.01153, 0.01673, 0.02177, 0.02725, 0.03228,
         0.03747, 0.04755, 0.06267, 0.08785, 0.13690,
         0.18476, 0.23084, 0.27588, 0.36601, 0.44750]
    )

    red_cal = LedCalibration("Red", currents, red_cal_p)
    green_cal = LedCalibration("Green", currents, green_cal_p)
    blue_cal = LedCalibration("Blue", currents, blue_cal_p)
    uv_cal = LedCalibration("UV", currents, uv_cal_p)

    # Bitmask definitions (LSB -> MSB)
    bitmasks = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=int)

    # Experiment-mode bit currents (%), in ascending bit order (1..128)
    red_curr = np.array([3.329, 4.004, 5.288, 7.807, 12.671, 22.763, 43.741, 99.876])
    green_curr = np.array([3.164, 3.664, 4.695, 6.757, 11.101, 20.587, 42.541, 99.788])
    blue_curr = np.array([3.114, 3.571, 4.501, 6.344, 10.251, 18.941, 39.835, 99.983])
    uv_curr = np.array([3.374, 4.097, 5.472, 8.142, 13.662, 24.894, 48.451, 99.975])

    # Experiment-mode powers per bit (µW), ascending bit order
    red_p_exp = np.array([0.40, 0.70, 1.40, 2.80, 5.50, 11.10, 22.50, 52.00])
    green_p_exp = np.array([0.20, 0.30, 0.50, 0.80, 1.50, 3.10, 6.50, 13.70])
    blue_p_exp = np.array([0.10, 0.10, 0.40, 0.70, 1.40, 2.90, 5.60, 11.30])
    uv_p_exp = np.array([0.10, 0.10, 0.20, 0.30, 0.60, 1.20, 2.30, 4.50])

    red_bits = BitmaskChannelData("Red", bitmasks, red_curr, red_p_exp)
    green_bits = BitmaskChannelData("Green", bitmasks, green_curr, green_p_exp)
    blue_bits = BitmaskChannelData("Blue", bitmasks, blue_curr, blue_p_exp)
    uv_bits = BitmaskChannelData("UV", bitmasks, uv_curr, uv_p_exp)

    # Linearize each channel (anchor MSB = last index)
    _, I_new_red, _ = linearize_led_bitmasks(red_cal, red_bits, anchor_index=-1)
    _, I_new_green, _ = linearize_led_bitmasks(green_cal, green_bits, anchor_index=-1)
    _, I_new_blue, _ = linearize_led_bitmasks(blue_cal, blue_bits, anchor_index=-1)
    _, I_new_uv, _ = linearize_led_bitmasks(uv_cal, uv_bits, anchor_index=-1)

    # --- Plot calibration fits for each LED ---
    plot_calibration_curve(red_cal)
    plot_calibration_curve(green_cal)
    plot_calibration_curve(blue_cal)
    plot_calibration_curve(uv_cal)

    # --- Plot bitmask powers and currents for each LED ---
    # Red
    P_target_red, _, _ = linearize_led_bitmasks(red_cal, red_bits, anchor_index=-1)
    plot_bitmask_powers(red_bits, P_target_red, title_suffix="(Red)")
    plot_bitmask_currents(red_bits, I_new_red, title_suffix="(Red)")

    # Green
    P_target_green, _, _ = linearize_led_bitmasks(green_cal, green_bits, anchor_index=-1)
    plot_bitmask_powers(green_bits, P_target_green, title_suffix="(Green)")
    plot_bitmask_currents(green_bits, I_new_green, title_suffix="(Green)")

    # Blue
    P_target_blue, _, _ = linearize_led_bitmasks(blue_cal, blue_bits, anchor_index=-1)
    plot_bitmask_powers(blue_bits, P_target_blue, title_suffix="(Blue)")
    plot_bitmask_currents(blue_bits, I_new_blue, title_suffix="(Blue)")

    # UV
    P_target_uv, _, _ = linearize_led_bitmasks(uv_cal, uv_bits, anchor_index=-1)
    plot_bitmask_powers(uv_bits, P_target_uv, title_suffix="(UV)")
    plot_bitmask_currents(uv_bits, I_new_uv, title_suffix="(UV)")

    # Finally, show all figures
    plt.show()

    # Print in RGB format (LED 4=Red, 3=Green, 2=Blue)
    print("=== RGB format (4=Red, 3=Green, 2=Blue) ===")
    print_rgb_table(I_new_red, I_new_green, I_new_blue)

    print()  # blank line

    # Print in GBUV format (LED 3=Green, 2=Blue, 1=UV)
    print("=== GBUV format (3=Green, 2=Blue, 1=UV) ===")
    print_gbuv_table(I_new_green, I_new_blue, I_new_uv)





