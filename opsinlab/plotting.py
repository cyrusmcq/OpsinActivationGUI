# opsinlab/plotting.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _annotate_matrix(ax, M, fmt="%.2e", color="white", fontsize=8):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt % M[i, j], ha='center', va='center', color=color, fontsize=fontsize)

def plot_activation_heatmap(ax, activation_matrix: pd.DataFrame):
    im = ax.imshow(activation_matrix.values, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(activation_matrix.columns)))
    ax.set_xticklabels(activation_matrix.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(activation_matrix.index)))
    ax.set_yticklabels(activation_matrix.index)
    ax.set_title('Activation (LEDs × Opsins)')
    _annotate_matrix(ax, activation_matrix.values, fmt="%.2e")
    return im

def plot_inverse_heatmap(ax, inverse_matrix, row_labels, col_labels):
    im = ax.imshow(inverse_matrix, aspect='auto', cmap='RdBu')
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_title('Inverse / Pseudoinverse')
    _annotate_matrix(ax, np.array(inverse_matrix), fmt="%.3f", color='black')
    return im

def plot_modulation_heatmap(ax, modulation_df: pd.DataFrame):
    im = ax.imshow(modulation_df.values, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xticks(range(len(modulation_df.columns)))
    ax.set_xticklabels(modulation_df.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(modulation_df.index)))
    ax.set_yticklabels(modulation_df.index)
    ax.set_title('Normalized LED Modulations')
    _annotate_matrix(ax, modulation_df.values, fmt="%.3f", color='black')
    return im

def plot_grouped_modulation_bars(ax, modulation_df: pd.DataFrame):
    M = modulation_df.T  # opsins × LEDs  → groups by opsin
    x = np.arange(len(M.index))
    w = min(0.8 / max(1, len(M.columns)), 0.25)
    for k, led in enumerate(M.columns):
        ax.bar(x + k*w, M[led].to_numpy(), width=w, label=led)
    ax.set_xticks(x + (len(M.columns)-1)*w/2)
    ax.set_xticklabels(M.index, rotation=45, ha='right')
    ax.set_ylabel('Normalized LED Modulation')
    ax.set_title('LED Modulations per Isolation')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_final_led_bars(ax, final_led_modulation, led_labels, limiting_opsin: str):
    vals = np.asarray(final_led_modulation)
    bars = ax.bar(led_labels, vals, alpha=0.7)
    ax.set_ylabel('LED Modulation %')
    ax.set_title(f'Final LED Modulation (limiting: {limiting_opsin})')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + (0.01 if v>=0 else -0.03), f"{v:.3f}", ha='center', va='bottom' if v>=0 else 'top')

def plot_summary_table(ax, matrix_results: dict, led_labels: list[str]):
    ax.axis('off')
    rows = [
        ['Overall Max Modulation:', f"{matrix_results['overall_max']:.4f}"],
        ['Limiting Opsin:', matrix_results['max_opsin']],
        ['', ''],
        ['Final LED Modulations:', ''],
    ]
    for led, mod in zip(led_labels, matrix_results.get('final_led_modulation', [])):
        rows.append([f'{led}:', f'{mod:.4f}'])
    table = ax.table(cellText=rows, cellLoc='left', loc='center', colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.3)
    ax.set_title('Summary', pad=12)
