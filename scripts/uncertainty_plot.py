"""
6-panel (r_p/(R1+R2), v_inf) heatmap of collAIder's deterministic-NN predictions.

Top row     : outcome classification, M_{1,final}, M_{2,final}.
Bottom row  : classification uncertainty, M_1 uncertainty (Bayesian-only),
              M_2 uncertainty (Bayesian-only).

The NN is queried directly across the full grid (bypassing the regime
classifier), following the approach of Paper_plots_classification.ipynb.
R1 and R2 still come from collAIder's StellarRadiusEstimator (POSYDON) but
only to normalize the x-axis.

Usage (from the collAIder repo root):
    python scripts/uncertainty_plot.py
Press Enter at each prompt to accept the default (M1=32, M2=32, age=0.001 Gyr).
"""

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Make ../src importable and chdir into it so the POSYDON relative path resolves.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)
os.chdir(SRC)

from model_NN import PerformCollision, EncounterRegimeClassifier  # noqa: E402


CLASS_COLORS = ['lightgrey', '#386FA4', '#678D58', '#8E3B46']  # 0 1 2 3
CLASS_NAMES = ['0 (no stars)', '1 (merger)', '2 (two stars)', '3 (stripped)']

# Axis ranges matching Elena's paper figures.
RP_FRAC_MIN, RP_FRAC_MAX = 0.1, 1.3
LOG_VINF_MIN, LOG_VINF_MAX = 0.77, 4.3
GRID_SIZE = 400


def ask(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return float(s) if s else float(default)


def load_nn_models():
    PerformCollision(age=0.001, pericenter=1.0, velocity_inf=10.0, mass1=1.0, mass2=1.0)
    return (
        PerformCollision._classification_model,
        PerformCollision._regression_model,
        PerformCollision._input_mean,
        PerformCollision._input_std,
    )


def run_grid(age, m1, m2, n):
    classifier = EncounterRegimeClassifier()
    r1 = float(classifier.StellarRadiusEstimator(age, m1))
    r2 = float(classifier.StellarRadiusEstimator(age, m2))
    r_sum = r1 + r2
    print(f"R1 = {r1:.3f} Rsun, R2 = {r2:.3f} Rsun, R1+R2 = {r_sum:.3f} Rsun")

    rp_frac = np.logspace(np.log10(RP_FRAC_MIN), np.log10(RP_FRAC_MAX), n)
    vinf = np.logspace(LOG_VINF_MIN, LOG_VINF_MAX, n)
    RPF, VINF = np.meshgrid(rp_frac, vinf)
    RP = RPF * r_sum

    rps = RP.ravel().astype(np.float32)
    vinfs = VINF.ravel().astype(np.float32)
    n_cells = rps.size

    class_model, reg_model, mean, std = load_nn_models()
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    X_phys = np.stack([
        np.full(n_cells, np.log10(age + 0.001), dtype=np.float32),
        np.log10(rps + 0.1),
        np.log10(vinfs + 10.0),
        np.full(n_cells, np.log(m1), dtype=np.float32),
        np.full(n_cells, np.log(m2), dtype=np.float32),
    ], axis=1)
    X_norm = torch.tensor((X_phys - mean) / std, dtype=torch.float32)

    print(f"Running NN on {n_cells} grid points...")
    with torch.no_grad():
        class_logits = class_model(X_norm)
        class_probs = torch.softmax(class_logits, dim=1).numpy()
        reg_softmax = reg_model(X_norm).numpy()
    print("Done.")

    pred_class = class_probs.argmax(axis=1)
    m_tot = m1 + m2
    M1 = reg_softmax[:, 0] * m_tot
    M2 = reg_softmax[:, 1] * m_tot

    mask_none = pred_class == 0
    mask_one = (pred_class == 1) | (pred_class == 3)
    M1 = np.where(mask_none, 0.0, M1)
    M2 = np.where(mask_none | mask_one, 0.0, M2)

    pred_class = pred_class.reshape(RPF.shape)
    M1 = M1.reshape(RPF.shape)
    M2 = M2.reshape(RPF.shape)
    class_probs = class_probs.reshape(RPF.shape + (4,))
    return rp_frac, vinf, pred_class, M1, M2, class_probs


def style_axes(ax):
    """Elena's convention: ticks pointing inward, with minor ticks on."""
    ax.set_xscale('log')
    ax.set_xlim(RP_FRAC_MIN, RP_FRAC_MAX)
    ax.set_ylim(LOG_VINF_MIN, LOG_VINF_MAX)
    ax.set_xticks([0.1, 1.0])
    ax.set_xticklabels([r'$0.1$', r'$1$'])
    ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ax.set_yticklabels([r'$1.0$', r'$1.5$', r'$2.0$', r'$2.5$', r'$3.0$', r'$3.5$', r'$4.0$'])
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', length=6, labelsize=11)
    ax.tick_params(axis='both', which='minor', direction='in', length=3)
    ax.set_xlabel(r'$r_p / (R_1 + R_2)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}\left( v_{\infty} \, [\rm km/s] \right)$', fontsize=14)


def plot_placeholder(ax, title):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               fill=True, facecolor='white', hatch='///',
                               edgecolor='lightgrey', alpha=0.6, lw=0))
    ax.text(0.5, 0.5, f'{title}\nAwaiting Bayesian backend',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=14, fontweight='bold', color='dimgrey')


def make_figure(m1, m2, age, rp_frac, vinf, pred_class, M1, M2, class_probs):
    log_v = np.log10(vinf)
    # 1 - p_max is at most 3/4 with 4 classes; the 4/3 factor rescales to [0, 1].
    class_uncert = (1.0 - class_probs.max(axis=-1)) * (4.0 / 3.0)

    class_cmap = mcolors.ListedColormap(CLASS_COLORS)
    class_norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)
    vmax_mass = float(m1 + m2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    im = axes[0, 0].pcolormesh(rp_frac, log_v, pred_class,
                                cmap=class_cmap, norm=class_norm, shading='auto')
    cb = fig.colorbar(im, ax=axes[0, 0], ticks=[0, 1, 2, 3])
    cb.set_ticklabels(CLASS_NAMES)
    axes[0, 0].set_title('Outcome classification', fontsize=16)

    im = axes[0, 1].pcolormesh(rp_frac, log_v, M1,
                                cmap='coolwarm', vmin=0, vmax=vmax_mass, shading='auto')
    fig.colorbar(im, ax=axes[0, 1], label=r'$M_\odot$')
    axes[0, 1].set_title(r'$M_{1,\rm final}$', fontsize=16)

    im = axes[0, 2].pcolormesh(rp_frac, log_v, M2,
                                cmap='coolwarm', vmin=0, vmax=vmax_mass, shading='auto')
    fig.colorbar(im, ax=axes[0, 2], label=r'$M_\odot$')
    axes[0, 2].set_title(r'$M_{2,\rm final}$', fontsize=16)

    im = axes[1, 0].pcolormesh(rp_frac, log_v, class_uncert,
                                cmap='magma', vmin=0, vmax=1, shading='auto')
    fig.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set_title('Classification uncertainty', fontsize=16)

    plot_placeholder(axes[1, 1], r'$M_1$ uncertainty')
    plot_placeholder(axes[1, 2], r'$M_2$ uncertainty')

    for ax in axes.ravel():
        style_axes(ax)
    for ax in axes[0, :]:
        ax.set_xlabel('')
    for ax in axes[:, 1:].ravel():
        ax.set_ylabel('')

    fig.suptitle(
        fr'$M_1 = {m1}\,M_\odot,\ M_2 = {m2}\,M_\odot,\ T = {age}\,\rm Gyr$',
        fontsize=16,
    )
    plt.tight_layout()
    return fig


def main():
    print("=" * 50)
    print("collAIder star collision plots")
    print("Enter values for star masses and stellar age")
    print("=" * 50)
    m1 = ask("Mass of star 1 (Msun)", 32.0)
    m2 = ask("Mass of star 2 (Msun)", 32.0)
    age = ask("Stellar age (Gyr)", 0.001)

    m1_run = max(m1, m2)
    m2_run = min(m1, m2)

    rp_frac, vinf, pred_class, M1, M2, class_probs = run_grid(age, m1_run, m2_run, GRID_SIZE)

    if m1 < m2:
        M1, M2 = M2, M1

    fig = make_figure(m1, m2, age, rp_frac, vinf, pred_class, M1, M2, class_probs)

    fig_dir = os.path.join(HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, f"uncertainty_plot_M1{m1}_M2{m2}_age{age}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved figure: {out}")
    plt.show()


if __name__ == "__main__":
    main()
