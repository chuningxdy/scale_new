import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============ STYLE CONFIGURATION ============
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts match Times New Roman

# ============ CONFIGURATION ============
INPUT_FILE = "/mfs1/u/chuning/scale_new/outputs/runs/ICML/2026-01-28-20-11_openwebtext2_pythia_BFGS_80PF/6_critical_batch_size/h_samples_with_nn_loss.csv"
#"/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-24-20-39_openwebtext2_pythia_adam_cosine_heldoutNBK/6_critical_batch_size/h_samples_with_nn_loss.csv"
OUTPUT_FILE = "/mfs1/u/chuning/scale_new/plot_NB_scatter"

# Constraint levels (4 levels each)
NK_LEVELS = [0.625e11, 1.25e11, 2.5e11, 5e11, 1e12]  # NK < t
NB_LEVELS = [2e9, 4e9, 8e9, 2e10, 6e10]  # NB < m (memory proxy)
D_LEVELS = [1.1e6,1.5e6, 2e6, 3e6, 6e6]  # D = BK < d (data constraint)

# ============ LOAD DATA ============
df = pd.read_csv(INPUT_FILE)

# Compute T = N * K for constraint
df['T'] = df['N'] * df['K']

# ============ PLOTTING ============
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(4, 1.5), constrained_layout=True)

# Add overall figure title
fig.suptitle('Scaling Laws When Two Resources Are Constrained', fontsize=10, y=1.15)

# Get unique K values
unique_K = sorted(df['K'].unique())

# Define plot limits
XLIM = [df['N'].min() * 0.8, df['N'].max() * 1.5]
YLIM = [df['B'].min() * 0.8, df['B'].max() * 1.5]

# Get compute budget C from the data (C = 6*N*B*K, approximately constant)
C_compute = 6 * df['N'].iloc[0] * df['B'].iloc[0] * df['K'].iloc[0]

# Default/reference config (red diamond)
DEFAULT_N = 64000000
DEFAULT_B = 816

# ============ PANEL 1: NK Constraints ============
ax = ax1

# Grey color palette
greys_nk = sns.color_palette("Greys", len(NK_LEVELS))

# Plot all data points
ax.scatter(df['N'], df['B'], s=2*2, fc="k", alpha=0.2, ec="none", zorder=0)

for j, t_level in enumerate(NK_LEVELS):
    # For NK = t_level with fixed compute C = 6NBK:
    # B = C / (6 * t_level), which is a horizontal line
    B_boundary = C_compute / (6 * t_level)

    # Plot horizontal line for the constraint boundary
    ax.axhline(y=B_boundary, color="k", linestyle='-', linewidth=1, zorder=0)

    # Shade the region OUTSIDE the constraint (NK >= t_level means B <= C/(6t))
    region = np.array([
        [XLIM[0], YLIM[0]],
        [XLIM[1], YLIM[0]],
        [XLIM[1], B_boundary],
        [XLIM[0], B_boundary]
    ])
    ax.fill(region[:, 0], region[:, 1], color="grey", alpha=0.08 * (j + 1), zorder=-1)

    # Filter points within constraint
    df_constrained = df[df['T'] < t_level].dropna(subset=['NN_loss', 'nqs_loss'])

    if len(df_constrained) > 0:
        min_nn_idx = df_constrained['NN_loss'].idxmin()
        min_nn_point = df_constrained.loc[min_nn_idx]
        min_nqs_idx = df_constrained['nqs_loss'].idxmin()
        min_nqs_point = df_constrained.loc[min_nqs_idx]

        nn_same_as_nqs = (min_nn_point['N'] == min_nqs_point['N']) and (min_nn_point['B'] == min_nqs_point['B'])

        if nn_same_as_nqs:
            ax.scatter(min_nqs_point['N'], min_nqs_point['B'], s=7*7, marker='+',
                       facecolor='b', edgecolor='b', zorder=2)
            ax.scatter(min_nn_point['N'], min_nn_point['B'], s=4*4, marker='o',
                       facecolor='k', edgecolor='k', zorder=1)
        else:
            ax.scatter(min_nqs_point['N'], min_nqs_point['B'], s=7*7, marker='+',
                       facecolor='b', edgecolor='b', zorder=1)
            ax.scatter(min_nn_point['N'], min_nn_point['B'], s=4*4, marker='o',
                       facecolor='k', edgecolor='k', zorder=1)

# Plot default config (red diamond)
ax.scatter(DEFAULT_N, DEFAULT_B, s=4*4, marker='D', facecolor='none', edgecolor='r', zorder=3)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
ax.set_xlabel('Model size $N$', fontsize=7, labelpad=1)
ax.set_ylabel('Batch size $B$', fontsize=7, labelpad=1)
#bold
ax.set_title('Compute & Time', fontsize=7, weight='bold')
ax.tick_params(axis='x', which='both', labelsize=7, pad=1)
ax.tick_params(axis='y', which='both', labelsize=7, pad=1)
ax.xaxis.set_major_locator(plt.LogLocator(base=10.0))
ax.yaxis.set_major_locator(plt.LogLocator(base=10.0))
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_box_aspect(1)

# ============ PANEL 2: Memory (NB) Constraints ============
ax = ax2

# Grey color palette
greys_nb = sns.color_palette("Greys", len(NB_LEVELS))

# Plot all data points
ax.scatter(df['N'], df['B'], s=2*2, fc="k", alpha=0.2, ec="none", zorder=0)

# Compute NB for filtering
df['NB'] = df['N'] * df['B']

for j, m_level in enumerate(NB_LEVELS):
    # For NB = m_level: B = m_level / N, which is a hyperbola

    # Create hyperbola boundary: B = m_level / N
    N_range = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 200)
    B_boundary = m_level / N_range

    # Clip to plot limits
    B_boundary = np.clip(B_boundary, YLIM[0], YLIM[1])

    # Plot hyperbola for the constraint boundary
    ax.plot(N_range, B_boundary, color="k", linestyle='-', linewidth=1, zorder=0)

    # Shade the region OUTSIDE the constraint (NB >= m_level means B >= m_level/N)
    # This is the region ABOVE the hyperbola
    region_N = np.concatenate([N_range, N_range[::-1]])
    region_B = np.concatenate([B_boundary, np.full_like(N_range, YLIM[1])])
    ax.fill(region_N, region_B, color="grey", alpha=0.08 * (j + 1), zorder=-1)

    # Filter points within constraint
    df_constrained = df[df['NB'] < m_level].dropna(subset=['NN_loss', 'nqs_loss'])

    if len(df_constrained) > 0:
        min_nn_idx = df_constrained['NN_loss'].idxmin()
        min_nn_point = df_constrained.loc[min_nn_idx]
        min_nqs_idx = df_constrained['nqs_loss'].idxmin()
        min_nqs_point = df_constrained.loc[min_nqs_idx]

        nn_same_as_nqs = (min_nn_point['N'] == min_nqs_point['N']) and (min_nn_point['B'] == min_nqs_point['B'])

        if nn_same_as_nqs:
            ax.scatter(min_nqs_point['N'], min_nqs_point['B'], s=7*7, marker='+',
                       facecolor='b', edgecolor='b', zorder=2)
            ax.scatter(min_nn_point['N'], min_nn_point['B'], s=4*4, marker='o',
                       facecolor='k', edgecolor='k', zorder=1)
        else:
            ax.scatter(min_nqs_point['N'], min_nqs_point['B'], s=7*7, marker='+',
                       facecolor='b', edgecolor='b', zorder=1)
            ax.scatter(min_nn_point['N'], min_nn_point['B'], s=4*4, marker='o',
                       facecolor='k', edgecolor='k', zorder=1)

# Plot default config (red diamond)
ax.scatter(DEFAULT_N, DEFAULT_B, s=4*4, marker='D', facecolor='none', edgecolor='r', zorder=3)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
ax.set_title('Compute & Memory', fontsize=7, weight='bold')
ax.tick_params(axis='x', which='both', labelsize=7, pad=1)
ax.tick_params(axis='y', which='both', labelsize=7, pad=1)
ax.xaxis.set_major_locator(plt.LogLocator(base=10.0))
ax.yaxis.set_major_locator(plt.LogLocator(base=10.0))
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_box_aspect(1)

# ============ PANEL 3: Data (D = BK) Constraints ============
ax = ax3

# Grey color palette
greys_d = sns.color_palette("Greys", len(D_LEVELS))

# Plot all data points
ax.scatter(df['N'], df['B'], s=2*2, fc="k", alpha=0.2, ec="none", zorder=0)

# Compute D = B * K for filtering
df['D'] = df['B'] * df['K']

for j, d_level in enumerate(D_LEVELS):
    # For D = d_level with fixed compute C = 6NBK:
    # D = BK, and C = 6NBK = 6ND, so N = C/(6D)
    # D < d_level means N > C/(6*d_level), which is a vertical line
    N_boundary = C_compute / (6 * d_level)

    # Plot vertical line for the constraint boundary
    ax.axvline(x=N_boundary, color="k", linestyle='-', linewidth=1, zorder=0)

    # Shade the region OUTSIDE the constraint (D >= d_level means N <= C/(6d))
    # This is the region to the LEFT of the vertical line
    region = np.array([
        [XLIM[0], YLIM[0]],
        [N_boundary, YLIM[0]],
        [N_boundary, YLIM[1]],
        [XLIM[0], YLIM[1]]
    ])
    ax.fill(region[:, 0], region[:, 1], color="grey", alpha=0.08 * (j + 1), zorder=-1)

    # Filter points within constraint
    df_constrained = df[df['D'] < d_level].dropna(subset=['NN_loss', 'nqs_loss'])

    if len(df_constrained) > 0:
        min_nn_idx = df_constrained['NN_loss'].idxmin()
        min_nn_point = df_constrained.loc[min_nn_idx]
        min_nqs_idx = df_constrained['nqs_loss'].idxmin()
        min_nqs_point = df_constrained.loc[min_nqs_idx]

        nn_same_as_nqs = (min_nn_point['N'] == min_nqs_point['N']) and (min_nn_point['B'] == min_nqs_point['B'])

        if nn_same_as_nqs:
            ax.scatter(min_nqs_point['N'], min_nqs_point['B'], s=7*7, marker='+',
                       facecolor='b', edgecolor='b', zorder=2)
            ax.scatter(min_nn_point['N'], min_nn_point['B'], s=4*4, marker='o',
                       facecolor='k', edgecolor='k', zorder=1)
        else:
            ax.scatter(min_nqs_point['N'], min_nqs_point['B'], s=7*7, marker='+',
                       facecolor='b', edgecolor='b', zorder=1)
            ax.scatter(min_nn_point['N'], min_nn_point['B'], s=4*4, marker='o',
                       facecolor='k', edgecolor='k', zorder=1)

# Plot default config (red diamond)
ax.scatter(DEFAULT_N, DEFAULT_B, s=4*4, marker='D', facecolor='none', edgecolor='r', zorder=3)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
ax.set_title('Compute & Data', fontsize=7, weight='bold')
ax.tick_params(axis='x', which='both', labelsize=7, pad=1)
ax.tick_params(axis='y', which='both', labelsize=7, pad=1)
ax.xaxis.set_major_locator(plt.LogLocator(base=10.0))
ax.yaxis.set_major_locator(plt.LogLocator(base=10.0))
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_box_aspect(1)

# ============ LEGEND ============
# Add legend entries (dummy plots outside visible range)
ax.scatter(10**12, 10**12, s=2*2, fc="k", alpha=0.2, ec="none", label="$(N,B,K)$ configs.")
ax.plot([10**12], [10**12], color="k", linewidth=1, linestyle='-', label="2nd resource boundary")
ax.fill([10**12, 10**12], [10**12, 10**12], color="grey", alpha=0.3, label="Out of boundary")
ax.scatter(10**12, 10**12, s=7*7, marker="+", facecolor="b", edgecolor='b', label="Opt. config. (NQS)")
ax.scatter(10**12, 10**12, s=4*4, marker="o", facecolor="k", edgecolor='k', label="Opt. config. (LLM)")
ax.scatter(10**12, 10**12, s=4*4, marker="D", facecolor="none", edgecolor='r', label="Default config.")

fig.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.02),
           frameon=False, handlelength=1, handletextpad=0.5, ncol=3)

plt.savefig(f"{OUTPUT_FILE}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig(f"{OUTPUT_FILE}.pdf", bbox_inches='tight', pad_inches=0.05)
plt.close()

print(f"Saved plot to {OUTPUT_FILE}.png and {OUTPUT_FILE}.pdf")
