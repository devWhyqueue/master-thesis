"""Schematic of the manifold-coverage metric used in the methods report.

Draws two training samples of equal size against one reference cloud with two
modes, so the panels differ only in how the training material is spread.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from sklearn.neighbors import NearestNeighbors

OUTPUT = Path(__file__).with_name("coverage_plot.png")
N_REF = 400
N_TRAIN = 40
K = 15
REFERENCE_COLOR = "#4ca3dd"
LABEL_BOX = {
    "boxstyle": "round,pad=0.3",
    "facecolor": "white",
    "alpha": 0.9,
    "edgecolor": "#cccccc",
}


def reference_radii(reference: np.ndarray, k: int) -> np.ndarray:
    """Distance from each reference point to its k-th nearest other reference point."""
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(reference)
    distances, _ = neighbors.kneighbors(reference)
    return distances[:, k]


def covered_mask(
    reference: np.ndarray, training: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    """Reference points holding at least one training point inside their own radius."""
    return np.array(
        [
            bool(np.any(np.linalg.norm(training - point, axis=1) < radius))
            for point, radius in zip(reference, radii)
        ]
    )


def _strip_axes(ax: Axes) -> None:
    """Remove ticks and spines so the panel reads as a schematic."""
    ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def _draw_reference(
    ax: Axes, reference: np.ndarray, radii: np.ndarray, covered: np.ndarray
) -> None:
    """Draw the reference cloud, splitting covered from uncovered points."""
    ax.scatter(
        *reference[~covered].T,
        c="#d3d3d3",
        label="Uncovered reference",
        alpha=0.7,
        s=15,
    )
    ax.scatter(
        *reference[covered].T,
        c=REFERENCE_COLOR,
        label="Covered reference",
        alpha=0.9,
        s=15,
    )
    for point, radius in zip(reference[covered], radii[covered]):
        circle = Circle(
            tuple(point), radius, color=REFERENCE_COLOR, fill=False, alpha=0.05
        )
        ax.add_patch(circle)


def plot_scenario(
    ax: Axes,
    reference: np.ndarray,
    training: np.ndarray,
    radii: np.ndarray,
    title: str,
) -> None:
    """Render one panel: reference cloud, training sample, and achieved coverage."""
    covered = covered_mask(reference, training, radii)
    _draw_reference(ax, reference, radii, covered)
    ax.scatter(
        *training.T,
        c="#d9381e",
        marker="X",
        label="Training (observed)",
        s=40,
        edgecolor="black",
        linewidths=0.5,
    )
    ax.set_title(title, pad=5, fontsize=11)
    ax.text(
        0.95,
        0.05,
        f"$\\mathrm{{cov}}_c = {covered.mean() * 100:.1f}$%",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="right",
        bbox=LABEL_BOX,
    )
    _strip_axes(ax)


def _sample_points(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference cloud plus two equally sized training samples spread differently."""
    reference = np.vstack(
        [
            rng.normal(size=(N_REF // 2, 2)) * 0.5 + [0, 0],
            rng.normal(size=(N_REF // 2, 2)) * 0.5 + [3, 2],
        ]
    )
    poor = rng.normal(size=(N_TRAIN, 2)) * 0.5 + [0, 0]
    good = np.vstack(
        [
            rng.normal(size=(N_TRAIN // 2, 2)) * 0.5 + [0, 0],
            rng.normal(size=(N_TRAIN // 2, 2)) * 0.5 + [3, 2],
        ]
    )
    return reference, poor, good


def _finish(fig: Figure, ax: Axes) -> None:
    """Add the shared legend and write the figure to disk."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(wspace=0.05, bottom=0.18, top=0.92, left=0.02, right=0.98)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    """Generate the two-panel coverage schematic."""
    rng = np.random.default_rng(42)
    reference, poor, good = _sample_points(rng)
    radii = reference_radii(reference, K)
    fig, (ax_poor, ax_good) = plt.subplots(1, 2, figsize=(8, 3.5), dpi=300)
    plot_scenario(ax_poor, reference, poor, radii, "(a) Poor coverage")
    plot_scenario(ax_good, reference, good, radii, "(b) Good coverage")
    _finish(fig, ax_poor)


def _self_check() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    radii = reference_radii(points, k=1)
    assert np.allclose(radii, 1.0), radii
    assert covered_mask(points, points, radii).all()
    assert not covered_mask(points, points + 100.0, radii).any()
    # Strict inequality: the neighbour lying exactly on the radius stays uncovered.
    one_point = covered_mask(points, points[:1], radii)
    assert one_point.tolist() == [True, False, False, False], one_point


if __name__ == "__main__":
    _self_check()
    main()
