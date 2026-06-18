"""
Batch folder analysis script for the ITHILDIN wing pipeline.

This script recursively scans a folder (including subfolders), runs the
ITHILDIN prediction pipeline on each image, stores per-image JSON outputs in a
newly created run folder, and exports two CSV files:
1. Raw landmark coordinates (before GPA)
2. GPA-aligned landmark coordinates (after Procrustes superimposition)
"""

import argparse
import os
from datetime import datetime

import pandas as pd

from analysis import landmark_analysis
from config_loader import AVAILABLE_FAMILIES
import main
import utils


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _is_image_file(filename):
	return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def _safe_output_stem(relative_path):
	stem = os.path.splitext(relative_path)[0]
	return stem.replace(os.sep, "__")


def _collect_images(root_folder):
	image_paths = []
	for current_root, _, files in os.walk(root_folder):
		for filename in files:
			if _is_image_file(filename):
				image_paths.append(os.path.join(current_root, filename))
	return sorted(image_paths)


def run_folder_pipeline(folder_path, family="mosquito"):
	"""
	Run the full ITHILDIN pipeline for all images in folder_path recursively.

	Args:
		folder_path (str): Base folder containing images in nested directories.
		family (str): Insect family key defined in config_loader.AVAILABLE_FAMILIES.

	Returns:
		dict: Paths to generated outputs and number of processed images.
	"""
	folder_path = os.path.abspath(folder_path)
	if not os.path.isdir(folder_path):
		raise FileNotFoundError(f"Folder not found: {folder_path}")

	if family not in AVAILABLE_FAMILIES:
		valid = ", ".join(sorted(AVAILABLE_FAMILIES.keys()))
		raise ValueError(f"Unknown family '{family}'. Available: {valid}")

	images = _collect_images(folder_path)
	if not images:
		raise ValueError(
			"No supported images found. Supported extensions: "
			f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
		)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir_name = f"ithildin_batch_{timestamp}"
	run_dir = os.path.join(folder_path, run_dir_name)
	os.makedirs(run_dir, exist_ok=False)

	print(f"Found {len(images)} images")
	print(f"Writing JSON outputs to: {run_dir}")

	for idx, image_path in enumerate(images, start=1):
		relative_path = os.path.relpath(image_path, folder_path)
		output_stem = _safe_output_stem(relative_path)
		save_path = os.path.join(run_dir, output_stem)

		# Ensure unique stem if sanitized names collide.
		counter = 1
		unique_save_path = save_path
		while os.path.exists(f"{unique_save_path}.json"):
			unique_save_path = f"{save_path}_{counter}"
			counter += 1

		image_id = f"{idx:05d}:{relative_path}"
		print(f"[{idx}/{len(images)}] Processing: {relative_path}")

		main.run_prediction(
			image_path,
			save_path=unique_save_path,
			family=family,
			stream=False,
			save_image=False,
			image_id=image_id,
			bg_session=None,
		)

	raw_csv_path = os.path.join(run_dir, "raw_landmarks.csv")
	gpa_csv_path = os.path.join(run_dir, "gpa_landmarks.csv")

	raw_df = utils.json_to_dataframe(
		run_dir,
		semilandmark=False,
		coordinate_type="unscaled",
		family=family,
		with_lm_predictions=False,
	)
	raw_df.to_csv(raw_csv_path, index=False, sep=";")

	gpa_meta_df, gpa_coords_df = landmark_analysis.procrustes(raw_df, semilandmark=False)
	gpa_df = pd.concat(
		[gpa_meta_df.reset_index(drop=True), gpa_coords_df.reset_index(drop=True)],
		axis=1,
	)
	gpa_df.to_csv(gpa_csv_path, index=False, sep=";")

	return {
		"run_dir": run_dir,
		"raw_csv": raw_csv_path,
		"gpa_csv": gpa_csv_path,
		"num_images": len(images),
	}


def main_cli():
	parser = argparse.ArgumentParser(
		description="Run ITHILDIN analysis for all images in a folder and subfolders."
	)
	parser.add_argument(
		"folder",
		help="Path to the folder containing images (searched recursively).",
	)
	parser.add_argument(
		"--family",
		default="mosquito",
		choices=sorted(AVAILABLE_FAMILIES.keys()),
		help="Insect family configuration to use.",
	)
	args = parser.parse_args()

	result = run_folder_pipeline(args.folder, family=args.family)
	print("\nBatch analysis complete")
	print(f"Processed images: {result['num_images']}")
	print(f"Run folder: {result['run_dir']}")
	print(f"Raw landmarks CSV: {result['raw_csv']}")
	print(f"GPA landmarks CSV: {result['gpa_csv']}")


if __name__ == "__main__":
	main_cli()
