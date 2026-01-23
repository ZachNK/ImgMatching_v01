import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
p = Path("/exports/figs/")
mu = 0
sigma = 1

x = np.linspace(-4, 4, 400)
y = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

plt.plot(x, y)
plt.title("Gaussian Distribution")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.grid(True)
plt.savefig(p / "figsgaussian.png", dpi=150)
plt.show()