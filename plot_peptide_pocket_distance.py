#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# 1) Load data (skip the header line that starts with '#')
data = np.loadtxt("com_distance.dat", comments="#")

time_ns = data[:, 0]
dist_A  = data[:, 1]

# 2) Make the plot
plt.figure()
plt.plot(time_ns, dist_A)
plt.xlabel("Time (ns)")
plt.ylabel("Peptide–pocket COM distance (Å)")
plt.title("COM distance vs time")
plt.tight_layout()

# 3) Save and/or show
plt.savefig("com_distance.png", dpi=300)
plt.show()
