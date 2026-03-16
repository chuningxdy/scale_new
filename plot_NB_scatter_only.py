import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ STYLE CONFIGURATION ============
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# ============ CONFIGURATION ============
INPUT_FILE = "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-24-20-39_openwebtext2_pythia_adam_cosine_heldoutNBK/6_critical_batch_size/h_samples_with_nn_loss.csv"
INPUT_FILE_LARGE_B = "/mfs1/u/chuning/scale_new/inputs/h_samples_test_nqs_owt_adam_cosine_NBK_corrected_very_large_B_add.csv"
INPUT_FILE_LARGE_K = "/mfs1/u/chuning/scale_new/inputs/h_samples_test_nqs_owt_adam_cosine_NBK_corrected_very_large_K_add.csv"
OUTPUT_FILE = "/mfs1/u/chuning/scale_new/plot_NB_scatter_only"

# ============ LOAD DATA ============
df = pd.read_csv(INPUT_FILE)
df_large_B = pd.read_csv(INPUT_FILE_LARGE_B)
df_large_K = pd.read_csv(INPUT_FILE_LARGE_K)

# ============ PLOTTING ============
fig, ax = plt.subplots(figsize=(4, 4))

# Plot original data in grey
ax.scatter(df['N'], df['B'], s=20, fc="grey", alpha=0.5, ec="none", zorder=1, label="Original")

# Plot large B data in red
ax.scatter(df_large_B['N'], df_large_B['B'], s=20, fc="red", alpha=0.7, ec="none", zorder=2, label="Large B")

# Plot large K data in blue
ax.scatter(df_large_K['N'], df_large_K['B'], s=20, fc="blue", alpha=0.7, ec="none", zorder=2, label="Large K")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('Model size $N$', fontsize=10)
ax.set_ylabel('Batch size $B$', fontsize=10)
ax.set_title('(N, B) Configurations', fontsize=12)

ax.legend(fontsize=8, loc='best', frameon=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FILE}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig(f"{OUTPUT_FILE}.pdf", bbox_inches='tight', pad_inches=0.05)
plt.close()

print(f"Saved plot to {OUTPUT_FILE}.png and {OUTPUT_FILE}.pdf")
