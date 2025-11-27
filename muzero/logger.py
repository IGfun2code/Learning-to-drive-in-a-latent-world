# muzero/logger.py

import os
import numpy as np
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, save_dir="logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.data = {}          # name -> list of (step, value)

    def log(self, name, step, value):
        """Record a scalar metric."""
        if name not in self.data:
            self.data[name] = []
        self.data[name].append((step, float(value)))

    def save_csv(self):
        """Write each metric to a CSV file."""
        for name, data in self.data.items():
            arr = np.array(data)
            np.savetxt(
                f"{self.save_dir}/{name}.csv", arr, delimiter=",",
                header="step,value", comments=""
            )

    def plot(self):
        """Plot each metric to its own PNG."""
        for name, data in self.data.items():
            arr = np.array(data)
            steps = arr[:, 0]
            values = arr[:, 1]

            plt.figure(figsize=(6,4))
            plt.plot(steps, values)
            plt.xlabel("Training step")
            plt.ylabel(name)
            plt.title(name)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/{name}.png", dpi=300)
            plt.close()

    def flush(self):
        """Save all plots & CSV files."""
        self.save_csv()
        self.plot()
