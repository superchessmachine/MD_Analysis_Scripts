#!/usr/bin/env python3
"""
example usage:
python plot_com_distance.py \
  --data com_distance.dat \
  --ymin 5 \
  --ymax 25 \
  --yinc 2
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot peptide–pocket COM distance vs time from a two-column file "
            "(time_ns, distance_A). Optionally fix y-axis range and tick spacing "
            "for normalized comparison across runs."
        )
    )

    p.add_argument(
        "--data",
        default="com_distance.dat",
        help="Input data file with columns: time_ns distance_A. Default: com_distance.dat",
    )
    p.add_argument(
        "--out",
        default="com_distance.png",
        help="Output image filename. Default: com_distance.png",
    )

    # Optional y-axis normalization controls
    p.add_argument(
        "--ymin",
        type=float,
        help="Fixed lower limit for y-axis (distance, Å). If provided with --ymax, y-limits will be fixed.",
    )
    p.add_argument(
        "--ymax",
        type=float,
        help="Fixed upper limit for y-axis (distance, Å). If provided with --ymin, y-limits will be fixed.",
    )
    p.add_argument(
        "--yinc",
        type=float,
        help=(
            "Tick spacing for y-axis (distance, Å). Only used if both --ymin and --ymax "
            "are also provided. Helps normalize plots for comparison."
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    # 1) Load data (skip the header line that starts with '#')
    data = np.loadtxt(args.data, comments="#")

    time_ns = data[:, 0]
    dist_A = data[:, 1]

    # 2) Make the plot
    plt.figure()
    plt.plot(time_ns, dist_A)
    plt.xlabel("Time (ns)")
    plt.ylabel("Peptide–pocket COM distance (Å)")
    plt.title("COM distance vs time")

    # Optional: fix y-axis range and tick spacing
    if args.ymin is not None and args.ymax is not None:
        plt.ylim(args.ymin, args.ymax)

        if args.yinc is not None and args.yinc > 0:
            yticks = np.arange(args.ymin, args.ymax + 1e-9, args.yinc)
            plt.yticks(yticks)

    plt.tight_layout()

    # 3) Save and/or show
    plt.savefig(args.out, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
