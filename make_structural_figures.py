"""Draw the two manuscript architecture figures as editable vector PDFs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#263d55"
BLUE = "#eaf1f8"
AMBER = "#fff2d9"
GREEN = "#e7f4eb"


def _box(ax, xy, size, title, detail, color=BLUE):
    x, y = xy
    w, h = size
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                linewidth=1.05, edgecolor=INK, facecolor=color))
    ax.text(x + 0.08, y + h - 0.13, title, va="top", ha="left",
            fontsize=8.2, fontweight="bold", color=INK)
    ax.text(x + 0.08, y + h - 0.41, detail, va="top", ha="left",
            fontsize=7.6, color=INK, linespacing=1.35)


def _arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.3, color=INK, shrinkA=0, shrinkB=0))


def build(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.05, 1.93))
    ax.set(xlim=(0, 10), ylim=(0, 2.55))
    ax.axis("off")
    for x, tag in [(0.17, "t0: DECISION START"), (3.54, "t1: URGENT ARRIVAL"),
                   (6.91, "t2: TX COMPLETE")]:
        ax.text(x, 2.36, tag, fontsize=8.1, fontweight="bold", color=INK)
    _box(ax, (.12, .48), (2.88, 1.61), "Idle link; queued traffic",
         "Snapshot s(t0); score a feasible\nnonurgent action. Pending token k;\n15 ms modeled decision delay.")
    _box(ax, (3.49, .48), (2.88, 1.61), "Arrival during inference",
         "Update live heap; cancel token k.\nStart urgent packet EDF with zero\nmodeled guard delay (idle link).", AMBER)
    _box(ax, (6.86, .48), (2.98, 1.61), "Transmission completes",
         "Log timely/late at true finish.\nRelease link; next decision uses\ncurrent queue and a new token.", GREEN)
    _arrow(ax, (3.04, 1.29), (3.44, 1.29))
    _arrow(ax, (6.41, 1.29), (6.81, 1.29))
    ax.text(5.0, .16, "Uncanceled slow decisions commit only at DECISION COMPLETE after live-queue validation.",
            fontsize=7.4, ha="center", color=INK)
    fig.subplots_adjust(left=.02, right=.99, top=.98, bottom=.01)
    fig.savefig(out_dir / "event_architecture_v2.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.05, 2.62))
    ax.set(xlim=(0, 10), ylim=(0, 3.5))
    ax.axis("off")
    _box(ax, (.10, 1.30), (2.08, 1.85), "Seven state features",
         "3 clipped queue counts\n3 mean remaining slacks\n1 capacity ratio")
    _box(ax, (.10, .08), (2.08, .93), "Candidate action", "2 angles: $0$ or $\\pi/2$", AMBER)
    _box(ax, (2.80, .63), (2.22, 2.20), "Nine input angles",
         "$x=[\\pi s_1,\\ldots,\\pi s_7,$\n$\\phi_1(a),\\phi_2(a)]$\n\nOne candidate per score", GREEN)
    _box(ax, (5.66, .63), (2.06, 2.20), "Nine-qubit circuit",
         "$R_x$ data injection\n$R_y$ + linear CNOT\n$R_y$ + linear CNOT")
    _box(ax, (8.32, .92), (1.56, 1.60), "Readout", "$\\langle Z_8\\rangle\\in[-1,1]$\n$Q(s,a)$; select\nfeasible argmax", AMBER)
    _arrow(ax, (2.22, 2.12), (2.75, 2.12))
    _arrow(ax, (2.22, .56), (2.75, 1.10))
    _arrow(ax, (5.06, 1.72), (5.61, 1.72))
    _arrow(ax, (7.76, 1.72), (8.27, 1.72))
    ax.text(5.0, 3.25, "Four action codes are distinct inputs; measured scores may still tie.",
            fontsize=7.8, ha="center", color=INK)
    fig.subplots_adjust(left=.02, right=.99, top=.98, bottom=.01)
    fig.savefig(out_dir / "joint_state_action_mapping.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build(Path(__file__).resolve().parent / "figures")
