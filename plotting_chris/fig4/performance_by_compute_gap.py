import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# ============ STYLE CONFIGURATION ============
# Modern, clean style for ML papers
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Colors
COLOR_TRAIN = 'blue'
COLOR_TEST = 'red'

# ============ CONFIGURATION ============
INPUT_FILE = "performance_by_compute_gap.csv"
OUTPUT_FILE = "performance_by_compute_gap_plot"

# ============ LOAD DATA ============
df = pd.read_csv(INPUT_FILE)

# Separate by model
df_chin = df[df['Model'] == 'Chin'].copy()
df_nqs = df[df['Model'] == 'NQS'].copy()

# Sort by Gap for proper line plotting
df_chin = df_chin.sort_values('Gap')
df_nqs = df_nqs.sort_values('Gap')

# ============ PLOTTING ============
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6, 3))

# Colors for series (only Train and IsoTest)
colors = {'Train': COLOR_TRAIN, 'IsoTest': COLOR_TEST}

# LEFT PANEL: Chinchilla
for col, color in colors.items():
    data = df_chin.dropna(subset=[col])
    if len(data) > 0:
        ax_left.scatter(data['Gap'], data[col], color=color, marker='o', s=50, alpha=0.8, ec='none')
        ax_left.plot(data['Gap'], data[col], color=color, linestyle='-', linewidth=3, alpha=0.7)

ax_left.set_xscale("log")
ax_left.set_yscale("log")
# ax_left.invert_xaxis()  # x-axis now: smallest to largest
ax_left.set_ylim(0.3, 15)
ax_left.set_xlim(0.5, 50000)
ax_left.set_xlabel("Compute Gap \n =   Max. Holdout FLOPs / Max. Train FLOPs", fontweight='bold')
ax_left.set_ylabel("Error", fontweight='bold')
ax_left.set_title("(a) Chinchilla", fontweight='bold')

# Set x-axis ticks: 1, 10, 100, 1000, 10000
ax_left.set_xticks([1, 10, 100, 1000, 10000])
ax_left.set_xticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
# Set y-axis ticks: 1, 10
ax_left.set_yticks([1, 10])
ax_left.set_yticklabels(['$10^0$', '$10^1$'])

ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)
ax_left.spines['left'].set_linewidth(2)
ax_left.spines['bottom'].set_linewidth(2)

# RIGHT PANEL: NQS
for col, color in colors.items():
    data = df_nqs.dropna(subset=[col])
    if len(data) > 0:
        ax_right.scatter(data['Gap'], data[col], color=color, marker='o', s=50, alpha=0.8, ec='none')
        ax_right.plot(data['Gap'], data[col], color=color, linestyle='-', linewidth=3, alpha=0.7)

ax_right.set_xscale("log")
ax_right.set_yscale("log")
# ax_right.invert_xaxis()  # x-axis now: smallest to largest
ax_right.set_ylim(0.3, 15)
ax_right.set_xlim(0.5, 50000)
ax_right.set_xlabel("Compute Gap", fontweight='bold')
# ax_right.set_ylabel("Error", fontweight='bold')
ax_right.set_title("(b) NQS", fontweight='bold')

# Set x-axis ticks: 1, 10, 100, 1000, 10000
ax_right.set_xticks([1, 10, 100, 1000, 10000])
ax_right.set_xticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
# Set y-axis ticks: 1, 10
ax_right.set_yticks([1, 10])
ax_right.set_yticklabels(['$10^0$', '$10^1$'])

ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.spines['left'].set_linewidth(2)
ax_right.spines['bottom'].set_linewidth(2)

# LEGEND
legend_elements = [
    Line2D([0], [0], marker='o', color=COLOR_TRAIN, linestyle='-', linewidth=3, markersize=8, alpha=0.7, label='Train'),
    Line2D([0], [0], marker='o', color=COLOR_TEST, linestyle='-', linewidth=3, markersize=8, alpha=0.7, label='Holdout'),
]

fig.legend(handles=legend_elements, fontsize=10, loc='lower center',
           bbox_to_anchor=(0.5, -0.12), ncol=2, prop={'weight': 'bold'})

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig(f"{OUTPUT_FILE}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_FILE}.pdf", bbox_inches='tight', pad_inches=0.0)
plt.close()

print(f"Saved plot to {OUTPUT_FILE}.png and {OUTPUT_FILE}.pdf")
