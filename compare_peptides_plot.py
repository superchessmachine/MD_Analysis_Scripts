#!/usr/bin/env python3
"""
Overlay COM distance traces from two simulations for comparison.

INPUT FORMAT
------------
Each input file must be a text file with:
    time_ns   distance_A
and may contain a header line starting with '#'.

EXAMPLE USAGE
-------------
python plot_com_overlay.py \
    --data1 peptide1/com_distance.dat \
    --data2 peptide2/com_distance.dat \
    --label1 "peptide1" \
    --label2 "peptide2" \
    --color1 "tab:blue" \
    --color2 "tab:orange" \
    --ymin 5 \
    --ymax 25 \
    --yinc 1 \
    --out com_overlay.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Overlay two peptide–pocket COM distance traces "
            "with customizable colors and labels."
        )
    )

    # Data + output
    p.add_argument(
        "--data1",
        required=True,
        help="First input data file (time_ns distance_A).",
    )
    p.add_argument(
        "--data2",
        required=True,
        help="Second input data file (time_ns distance_A).",
    )
    p.add_argument(
        "--out",
        default="com_overlay.png",
        help="Output image filename. Default: com_overlay.png",
    )

    # Labels
    p.add_argument(
        "--label1",
        default="Trace 1",
        help="Legend label for first dataset. Default: 'Trace 1'.",
    )
    p.add_argument(
        "--label2",
        default="Trace 2",
        help="Legend label for second dataset. Default: 'Trace 2'.",
    )

    # Colors
    p.add_argument(
        "--color1",
        default="tab:blue",
        help="Matplotlib color for first dataset. Default: 'tab:blue'.",
    )
    p.add_argument(
        "--color2",
        default="tab:orange",
        help="Matplotlib color for second dataset. Default: 'tab:orange'.",
    )

    # Optional y-axis normalization controls
    p.add_argument(
        "--ymin",
        type=float,
        help="Fixed lower limit for y-axis (distance, Å).",
    )
    p.add_argument(
        "--ymax",
        type=float,
        help="Fixed upper limit for y-axis (distance, Å).",
    )
    p.add_argument(
        "--yinc",
        type=float,
        help=(
            "Tick spacing for y-axis (distance, Å). Only used if both "
            "--ymin and --ymax are provided."
        ),
    )

    # Optional title override
    p.add_argument(
        "--title",
        default="COM distance vs time (overlay)",
        help="Plot title. Default: 'COM distance vs time (overlay)'.",
    )

    return p.parse_args()


def load_data(path):
    data = np.loadtxt(path, comments="#")
    time_ns = data[:, 0]
    dist_A = data[:, 1]
    return time_ns, dist_A


def main():
    args = parse_args()

    # Load both datasets
    t1, d1 = load_data(args.data1)
    t2, d2 = load_data(args.data2)

    plt.figure()

    # Plot both traces
    plt.plot(t1, d1, label=args.label1, color=args.color1)
    plt.plot(t2, d2, label=args.label2, color=args.color2)

    plt.xlabel("Time (ns)")
    plt.ylabel("Peptide–pocket COM distance (Å)")
    plt.title(args.title)

    # Y-axis normalization (optional)
    if args.ymin is not None and args.ymax is not None:
        plt.ylim(args.ymin, args.ymax)

        if args.yinc is not None and args.yinc > 0:
            yticks = np.arange(args.ymin, args.ymax + 1e-9, args.yinc)
            plt.yticks(yticks)

    plt.legend()
    plt.tight_layout()

    plt.savefig(args.out, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
