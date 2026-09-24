from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
SINE_TRAJECTORY_PATH = (
	Path(__file__).resolve().parents[2]
	/ "simulation"
	/ "fig2"
	/ "H_trajectory_920_sine.npy"
)
CONSTANT_TRAJECTORY_PATH = (
	Path(__file__).resolve().parents[2]
	/ "simulation"
	/ "H_trajectory_1500.npy"
)


def load_cycle_values(trajectory_path, n_cycles=100):
	"""Load flattened [H_mid, H_high, H_low] data and reshape it by cycle."""
	trajectory = np.asarray(np.load(trajectory_path), dtype=float).ravel()
	if trajectory.size < 3:
		raise ValueError("H trajectory must contain at least one three-point cycle")

	usable_size = trajectory.size - trajectory.size % 3
	cycle_values = trajectory[:usable_size].reshape(-1, 3)
	return cycle_values[:n_cycles]


def plot_trajectory_data(trajectory_path, output_path, n_cycles=100):
	"""Plot the first N cycles of a flattened trajectory without arrows."""
	cycle_values = load_cycle_values(trajectory_path, n_cycles)
	cycle = np.arange(1, len(cycle_values) + 1)

	fig, ax = plt.subplots(figsize=(10, 4.8))
	ax.plot(cycle, cycle_values[:, 0], color="black", lw=1.5, label="H_mid")
	ax.plot(cycle, cycle_values[:, 1], color="black", lw=1.5, label="H_high")
	ax.plot(cycle, cycle_values[:, 2], color="#dc2626", lw=2.0, label="H_low")

	ax.set_xlabel("N (cycle number)")
	ax.set_ylabel("Magnetic field H (mT)")
	ax.set_title(f"H trajectory data: N = 1-{len(cycle_values)}")
	ax.grid(alpha=0.25)
	ax.legend(ncol=3)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def plot_trajectory_structure(trajectory_path, output_path, n_cycles=4):
	"""Show the time order with t on the bottom and N on the top axis."""
	cycle_values = load_cycle_values(trajectory_path, n_cycles)
	input_time = np.arange(3 * len(cycle_values))
	values = cycle_values.ravel()

	fig, ax = plt.subplots(figsize=(11, 5.2))
	ax.plot(input_time, values, color="black", lw=1.2, alpha=0.75)
	ax.scatter(input_time[2::3], values[2::3], color="#dc2626", s=42, zorder=3, label="H_low")
	ax.scatter(input_time[0::3], values[0::3], color="black", s=28, zorder=3, label="H_mid / H_high")
	ax.scatter(input_time[1::3], values[1::3], color="black", s=28, zorder=3)

	for index in range(len(values) - 1):
		ax.annotate(
			"",
			xy=(input_time[index + 1], values[index + 1]),
			xytext=(input_time[index], values[index]),
			arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.0},
		)

	ax.set_xlabel("Input time series index t")
	ax.set_ylabel("Magnetic field H (mT)")
	ax.set_title("Input order: H_mid -> H_high -> H_low -> next H_mid")
	ax.set_xticks(input_time)
	ax.set_xticklabels([str(index) for index in input_time])
	ax.grid(alpha=0.25)

	top_axis = ax.twiny()
	top_axis.set_xlim(ax.get_xlim())
	cycle_positions = np.arange(1, len(cycle_values) + 1) * 3 - 2
	top_axis.set_xticks(cycle_positions)
	top_axis.set_xticklabels([f"N={index}" for index in range(1, len(cycle_values) + 1)])
	top_axis.set_xlabel("N (cycle number)")

	ax.legend(ncol=2, loc="best")
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def main():
	plot_trajectory_data(
		SINE_TRAJECTORY_PATH,
		OUTPUT_DIR / "H_trajectory_sine_N100.png",
	)
	plot_trajectory_data(
		CONSTANT_TRAJECTORY_PATH,
		OUTPUT_DIR / "H_trajectory_1500_N100.png",
	)
	plot_trajectory_structure(
		SINE_TRAJECTORY_PATH,
		OUTPUT_DIR / "H_trajectory_structure.png",
	)
	print("Saved H_trajectory_sine_N100.png, H_trajectory_1500_N100.png, and H_trajectory_structure.png")


if __name__ == "__main__":
	main()
