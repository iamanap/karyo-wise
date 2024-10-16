from pathlib import Path
import numpy as np
import torch
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import typer
from loguru import logger
from tqdm import tqdm

import karyo_wise.config as config

app = typer.Typer()


@app.command()
def main(
	# ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
	input_path: Path = config.PROCESSED_DATA_DIR / "dataset.csv",
	output_path: Path = config.FIGURES_DIR / "plot.png",
	# -----------------------------------------
):
	# ---- REPLACE THIS WITH YOUR OWN CODE ----
	logger.info("Generating plot from data...")
	for i in tqdm(range(10), total=10):
									if i == 5:
																	logger.info("Something happened for iteration 5.")
	logger.success("Plot generation complete.")
	# -----------------------------------------

	
def show_mask(mask, ax, random_color=False):
	if random_color:
			color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
	else:
			color = np.array([30/255, 144/255, 255/255, 0.6])
	h, w = mask.shape[-2:]
	mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
	ax.imshow(mask_image)
	
def show_box(box, ax):
	x0, y0 = box[0], box[1]
	w, h = box[2] - box[0], box[3] - box[1]
	ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_masks_and_boxes(img, masks, boxes):
	n_masks, n_predictions, h, w = masks.shape
	fig, axes = plt.subplots(1, n_predictions, figsize=(15, 5))

	if n_predictions == 1:
		axes = [axes]

	for i, ax in enumerate(axes):
		ax.imshow(img)
		specific_mask = masks[:, i]

		for j, mask in enumerate(specific_mask):
			show_mask(mask.cpu().numpy(), ax, random_color=True)
			if boxes is not None and j < len(boxes):
					show_box(boxes[j], ax)

		ax.axis('off')

	plt.tight_layout()
	plt.show()

def plot_label_distributions(save_to_reports=False):
	df_train = pd.read_csv(config.TRAIN_DATA_CSV)
	df_train['Label'] = df_train['Label'].astype(int)
	df_test = pd.read_csv(config.TEST_DATA_CSV)
	df_test['Label'] = df_test['Label'].astype(int)

	fig, axes = plt.subplots(2, 1, figsize=(15, 5))
	sns.countplot(x='Label', data=df_train, 
							order=sorted(df_train['Label'].unique()), 
							palette='Set2', hue="Label", legend=False,
							ax=axes[0])
	axes[0].set_xlabel('Label')
	axes[0].set_ylabel('Count')
	axes[0].set_title('Distribution of Labels - Train Dataset')

	sns.countplot(x='Label', data=df_test, 
							order=sorted(df_test['Label'].unique()), 
							palette='Set2', hue="Label", legend=False,
							ax=axes[1])
	axes[1].set_xlabel('Label')
	axes[1].set_ylabel('Count')
	axes[1].set_title('Distribution of Labels - Test Dataset')

	plt.tight_layout()
	plt.show()

	if save_to_reports:
			plt.savefig(config.FIGURES_DIR / "label_distribution.png")

if __name__ == "__main__":
	app()
