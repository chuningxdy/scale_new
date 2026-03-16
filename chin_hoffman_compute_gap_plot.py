import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============ STYLE CONFIGURATION ============
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Colors
COLOR_TRAIN = 'blue'
COLOR_TEST = 'red'

# ============ CONFIGURATION ============
INPUT_FILE = "/mfs1/u/chuning/scale_new/chin_rebuttal_hoffman/all_results.csv"
OUTPUT_FILE = "/mfs1/u/chuning/scale_new/chin_hoffman_compute_gap_plot"

# ============ LOAD DATA ============
df = pd.read_csv(INPUT_FILE)

# IsoFLOPs used in the sweep (in order)
isoflops = [5.5e18, 9.7e18, 2.8e19, 5.6e19, 8.7e19, 2.8e20, 5.6e20, 1e21, 3e21]

# Test FLOPs are always [1e21, 3e21], max is 3e21
max_test_flop = 3e21

# Compute the compute gap for each train_slice_count
# train_flops = isoflops[:train_slice_count], so max_train_flop = isoflops[train_slice_count - 1]
df['max_train_flop'] = df['train_slice_count'].apply(lambda x: isoflops[x - 1])
df['Gap'] = max_test_flop / df['max_train_flop']

# Exclude compute gap = 1.0 (train_slice_count=9, missing test evaluation)
df = df[df['Gap'] > 1.0]

# Use Huber Loss as the error metric, scaled by 10^5
df['Train'] = df['train_huber'] * 1e5
df['Test'] = df['test_huber_pure'] * 1e5

# Sort by Gap for proper line plotting
df = df.sort_values('Gap')

# ============ PLOTTING ============
fig, ax = plt.subplots(figsize=(3, 3))

# Colors for series
colors = {'Train': COLOR_TRAIN, 'Test': COLOR_TEST}

# Plot each series
for col, color in colors.items():
    data = df.dropna(subset=[col])
    if len(data) > 0:
        ax.scatter(data['Gap'], data[col], color=color, marker='o', s=50, alpha=0.8, ec='none')
        ax.plot(data['Gap'], data[col], color=color, linestyle='-', linewidth=3, alpha=0.7)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(2, 500)
ax.set_ylim(0.5, 10)
ax.set_xlabel("Compute Gap", fontweight='bold')
ax.set_ylabel("Error", fontweight='bold')
ax.set_title("Chinchilla on Hoffman Data", fontweight='bold',
                pad=20)

# Add scale indicator at top of y-axis
ax.text(-0.02, 1.02, r'$\times 10^{-5}$', transform=ax.transAxes, fontsize=10, va='bottom', ha='right')

# Set x-axis ticks: 1, 10, 100
ax.set_xticks([10, 100])
ax.set_xticklabels([ '$10^1$', '$10^2$'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# LEGEND
legend_elements = [
    Line2D([0], [0], marker='o', color=COLOR_TRAIN, linestyle='-', linewidth=3, markersize=8, alpha=0.7, label='Train'),
    Line2D([0], [0], marker='o', color=COLOR_TEST, linestyle='-', linewidth=3, markersize=8, alpha=0.7, label='Test'),
]

fig.legend(handles=legend_elements, fontsize=10, loc='lower center',
           bbox_to_anchor=(0.5, -0.06), ncol=2, prop={'weight': 'bold'})

plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
plt.savefig(f"{OUTPUT_FILE}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_FILE}.pdf", bbox_inches='tight', pad_inches=0.0)
plt.close()

print(f"Saved plot to {OUTPUT_FILE}.png and {OUTPUT_FILE}.pdf")

# Print the data for reference
print("\nData used (Huber Loss x 10^5):")
print(df[['train_slice_count', 'Gap', 'Train', 'Test']].to_string(index=False))
