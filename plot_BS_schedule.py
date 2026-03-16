import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ STYLE CONFIGURATION ============
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# ============ CONFIGURATION ============
INPUT_FILE = "outputs/runs/Paper/2026-01-26-16-42_openwebtext2_pythia_adam_cosine/6_critical_batch_size/h_samples_with_nn_loss.csv"
OUTPUT_FILE = "/mfs1/u/chuning/scale_new/plot_BS_schedule2"

# ============ LOAD DATA ============
df = pd.read_csv(INPUT_FILE)

# Get unique step_decay_schedules
unique_schedules = df['step_decay_schedule'].unique()

# Create color map
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_schedules)))
schedule_colors = {sched: colors[i] for i, sched in enumerate(unique_schedules)}

# Create short labels for schedules
def get_short_label(schedule):
    if 'B_decay_amt' not in schedule:
        return 'No B schedule'
    elif '[2.0, 4.0]' in schedule or '[2.0,4.0]' in schedule:
        return 'B increase 2x,4x'
    elif '[0.5, 0.25]' in schedule or '[0.5,0.25]' in schedule:
        return 'B decrease 0.5x,0.25x'
    elif '[3.0]' in schedule:
        return 'B increase 3x'
    elif '[0.33333]' in schedule:
        return 'B decrease 0.33x'
    elif '[2.0, 4.0, 8.0, 16.0]' in schedule or '[2.0,4.0,8.0,16.0]' in schedule:
        return 'B increase 2x,4x,8x,16x'
    else:
        return schedule[:30] + '...'

# ============ PLOTTING ============
fig, ax = plt.subplots(figsize=(6, 4))

for schedule in unique_schedules:
    df_sched = df[df['step_decay_schedule'] == schedule].sort_values('K')
    color = schedule_colors[schedule]
    label = get_short_label(schedule)

    # Plot nqs_loss as lines
    ax.plot(df_sched['K'], df_sched['nqs_loss'], color=color, linestyle='-', linewidth=2, alpha=0.7)

    # Plot NN_loss as dots
    ax.scatter(df_sched['K'], df_sched['NN_loss'], color=color, s=40, alpha=0.8, ec='none', label=label)

ax.set_xscale("log")
ax.invert_xaxis()  # Reverse order: large K to small K
ax.set_xlabel('Steps $K$', fontsize=10)
ax.set_ylabel('Loss', fontsize=10)
ax.set_title('Loss vs Steps by Schedule', fontsize=12)

# Add legend explaining dots vs lines
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], marker='o', color='grey', linestyle='', markersize=6, label='NN Loss (dots)'),
    Line2D([0], [0], color='grey', linestyle='-', linewidth=2, label='NQS Loss (lines)')
]

# First legend for dots vs lines
leg1 = ax.legend(handles=custom_lines, loc='upper left', fontsize=8, frameon=False)
ax.add_artist(leg1)

# Second legend for schedules
ax.legend(loc='upper right', fontsize=7, frameon=False, title='Schedule', title_fontsize=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FILE}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig(f"{OUTPUT_FILE}.pdf", bbox_inches='tight', pad_inches=0.05)
plt.close()

print(f"Saved plot to {OUTPUT_FILE}.png and {OUTPUT_FILE}.pdf")

# ============ SECOND PLOT: RANK OF LOSS (TWO PANELS) ============
# Compute ranks within each K value (1 = lowest/best, higher = worse)
df['nqs_rank'] = df.groupby('K')['nqs_loss'].rank(method='min')
df['NN_rank'] = df.groupby('K')['NN_loss'].rank(method='min')

# Define schedule order from dark to light
schedule_order_labels = ['2x,4x', '2x,4x,8x,16x', '3x', 'no sch', '0.33x', '0.5x,0.25x']

# Map schedules to their order
def get_schedule_order(schedule):
    if 'B_decay_amt' not in schedule:
        return 3  # no sch
    elif '[2.0, 4.0, 8.0, 16.0]' in schedule or '[2.0,4.0,8.0,16.0]' in schedule:
        return 1  # 2x,4x,8x,16x
    elif '[2.0, 4.0]' in schedule or '[2.0,4.0]' in schedule:
        return 0  # 2x,4x
    elif '[3.0]' in schedule:
        return 2  # 3x
    elif '[0.33333]' in schedule:
        return 4  # 0.33x
    elif '[0.5, 0.25]' in schedule or '[0.5,0.25]' in schedule:
        return 5  # 0.5x,0.25x
    else:
        return 6

# Sort schedules by order
sorted_schedules = sorted(unique_schedules, key=get_schedule_order)

# Create color scale from dark to light
n_schedules = len(sorted_schedules)
rank_colors = plt.cm.Blues(np.linspace(0.8, 0.2, n_schedules))  # dark to light
schedule_rank_colors = {sched: rank_colors[i] for i, sched in enumerate(sorted_schedules)}

# Short labels for legend
def get_short_label_rank(schedule):
    if 'B_decay_amt' not in schedule:
        return 'No schedule'
    elif '[2.0, 4.0, 8.0, 16.0]' in schedule or '[2.0,4.0,8.0,16.0]' in schedule:
        return '2x,4x,8x,16x'
    elif '[2.0, 4.0]' in schedule or '[2.0,4.0]' in schedule:
        return '2x,4x'
    elif '[3.0]' in schedule:
        return '3x'
    elif '[0.33333]' in schedule:
        return '0.33x'
    elif '[0.5, 0.25]' in schedule or '[0.5,0.25]' in schedule:
        return '0.5x,0.25x'
    else:
        return schedule[:20] + '...'

fig2, (ax_nqs, ax_llm) = plt.subplots(1, 2, figsize=(10, 4))

# Left panel: NQS
for schedule in sorted_schedules:
    df_sched = df[df['step_decay_schedule'] == schedule].sort_values('K')
    color = schedule_rank_colors[schedule]
    label = get_short_label_rank(schedule)
    ax_nqs.plot(df_sched['K'], df_sched['nqs_rank'], color=color, linestyle='-', linewidth=2, alpha=0.9, label=label)

ax_nqs.set_xscale("log")
ax_nqs.invert_xaxis()
ax_nqs.set_xlabel('Steps $K$', fontsize=10)
ax_nqs.set_ylabel('Rank (1 = best)', fontsize=10)
ax_nqs.set_title('NQS Loss Rank', fontsize=12)
ax_nqs.set_yticks(range(1, int(df['nqs_rank'].max()) + 1))
ax_nqs.invert_yaxis()  # Lowest rank (best) on top
ax_nqs.legend(loc='best', fontsize=7, frameon=False, title='B Schedule', title_fontsize=8)
ax_nqs.spines['top'].set_visible(False)
ax_nqs.spines['right'].set_visible(False)

# Right panel: LLM
for schedule in sorted_schedules:
    df_sched = df[df['step_decay_schedule'] == schedule].sort_values('K')
    color = schedule_rank_colors[schedule]
    label = get_short_label_rank(schedule)
    ax_llm.plot(df_sched['K'], df_sched['NN_rank'], color=color, linestyle='-', linewidth=2, alpha=0.9, label=label)

ax_llm.set_xscale("log")
ax_llm.invert_xaxis()
ax_llm.set_xlabel('Steps $K$', fontsize=10)
ax_llm.set_ylabel('Rank (1 = best)', fontsize=10)
ax_llm.set_title('LLM Loss Rank', fontsize=12)
ax_llm.set_yticks(range(1, int(df['NN_rank'].max()) + 1))
ax_llm.invert_yaxis()  # Lowest rank (best) on top
ax_llm.legend(loc='best', fontsize=7, frameon=False, title='B Schedule', title_fontsize=8)
ax_llm.spines['top'].set_visible(False)
ax_llm.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FILE}_rank.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig(f"{OUTPUT_FILE}_rank.pdf", bbox_inches='tight', pad_inches=0.05)
plt.close()

print(f"Saved rank plot to {OUTPUT_FILE}_rank.png and {OUTPUT_FILE}_rank.pdf")
