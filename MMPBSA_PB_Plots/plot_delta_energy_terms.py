"""
Generate plots and summary stats from GMX_MMPBSA delta energy terms.

Outputs:
- delta_TOTAL_vs_frame.png
- delta_gas_vs_solv.png
- delta_components_vs_frame.png
- delta_TOTAL_hist.png
- delta_TOTAL_running_avg.png
"""
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for scriptable runs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
DELTA_FILE = DATA_DIR / "delta_energy_terms.csv"


def load_delta_energy(path: Path) -> pd.DataFrame:
    """Load, clean, and sort the delta energy terms CSV."""
    df = pd.read_csv(path)
    df = df.dropna(how="all")

    numeric_cols = [c for c in df.columns if c != "Frame #"]
    df["Frame #"] = pd.to_numeric(df["Frame #"], errors="coerce")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Frame #"])
    df = df.sort_values("Frame #").reset_index(drop=True)
    return df


def gaussian_kde(values: np.ndarray, grid_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Simple Gaussian KDE without external dependencies."""
    arr = np.asarray(values)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])

    std = arr.std(ddof=1)
    bandwidth = 1.06 * std * arr.size ** (-1 / 5) if std > 0 else 1.0
    bandwidth = bandwidth if np.isfinite(bandwidth) and bandwidth > 0 else 1.0

    grid = np.linspace(arr.min(), arr.max(), grid_points)
    diff = grid[:, None] - arr[None, :]
    density = np.exp(-0.5 * (diff / bandwidth) ** 2).sum(axis=1)
    density /= arr.size * bandwidth * np.sqrt(2 * np.pi)
    return grid, density


def save_plot(fig: plt.Figure, filename: str) -> None:
    output_path = DATA_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {output_path.name}")


def plot_totals(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Frame #"], df["TOTAL"], color="tab:red", linewidth=1.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("ΔTOTAL (kcal/mol)")
    ax.set_title("Binding ΔTOTAL vs Frame")
    ax.grid(True, alpha=0.3)
    save_plot(fig, "delta_TOTAL_vs_frame.png")


def plot_gas_vs_solv(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Frame #"], df["GGAS"], label="ΔGGAS", color="tab:blue", linewidth=1.5)
    ax.plot(df["Frame #"], df["GSOLV"], label="ΔGSOLV", color="tab:green", linewidth=1.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Energy (kcal/mol)")
    ax.set_title("ΔGGAS and ΔGSOLV vs Frame")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "delta_gas_vs_solv.png")


def plot_components(df: pd.DataFrame, component_cols) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in component_cols:
        ax.plot(df["Frame #"], df[col], label=f"Δ{col}", linewidth=1.0)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Energy (kcal/mol)")
    ax.set_title("Δ Energy Components vs Frame")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    save_plot(fig, "delta_components_vs_frame.png")


def plot_total_hist(df: pd.DataFrame) -> None:
    vals = df["TOTAL"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=30, density=True, color="lightgray", edgecolor="black", alpha=0.7, label="Histogram")

    grid, density = gaussian_kde(vals.to_numpy())
    if grid.size > 0:
        ax.plot(grid, density, color="tab:red", linewidth=2, label="KDE")

    ax.set_xlabel("ΔTOTAL (kcal/mol)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of ΔTOTAL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "delta_TOTAL_hist.png")


def plot_running_average(df: pd.DataFrame, window: int = 10) -> None:
    rolling = df["TOTAL"].rolling(window=window, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Frame #"], df["TOTAL"], color="lightgray", linewidth=1.0, label="ΔTOTAL")
    ax.plot(df["Frame #"], rolling, color="tab:purple", linewidth=2, label=f"Running Avg (window={window})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("ΔTOTAL (kcal/mol)")
    ax.set_title("ΔTOTAL Running Average")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "delta_TOTAL_running_avg.png")


def summary_statistics(df: pd.DataFrame, cols) -> Dict[str, Dict[str, float]]:
    stats = df[cols].agg(["mean", "std", "min", "max"]).to_dict()
    print("\nSummary statistics (kcal/mol):")
    for col in cols:
        col_stats = stats[col]
        print(
            f"{col:>10} | mean: {col_stats['mean']:.2f} | "
            f"std: {col_stats['std']:.2f} | min: {col_stats['min']:.2f} | max: {col_stats['max']:.2f}"
        )
    return stats


def main() -> None:
    if not DELTA_FILE.exists():
        raise FileNotFoundError(f"Missing delta energy terms file: {DELTA_FILE}")

    df = load_delta_energy(DELTA_FILE)
    component_cols = ["VDWAALS", "EEL", "EPB", "ENPOLAR", "EDISPER"]
    stats_cols = component_cols + ["GGAS", "GSOLV", "TOTAL"]

    plot_totals(df)
    plot_gas_vs_solv(df)
    plot_components(df, component_cols)
    plot_total_hist(df)
    plot_running_average(df, window=10)
    summary_statistics(df, stats_cols)


if __name__ == "__main__":
    main()
