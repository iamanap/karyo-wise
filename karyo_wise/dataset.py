from pathlib import Path
from typing import List, Tuple

import typer
from loguru import logger
from tqdm import tqdm
import xml.etree.ElementTree as ET

from karyo_wise.config import RAW_DATA_DIR_IMG, RAW_DATA_DIR_ANN

app = typer.Typer()

def segment_images(input_img_path: Path, input_ann_path: Path):
    logger.info("Segmenting images...")


    logger.success("Image segmentation complete.")
    print('ok')

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_img_path: Path = RAW_DATA_DIR_IMG,
    input_ann_path: Path = RAW_DATA_DIR_ANN,
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------

def parse_voc_xml_to_bounding_box(xml_file: str) -> List:
    """
    Parses a Pascal VOC XML file and extracts bounding box coordinates.

    Arguments:
      xml_file (str): Path to the XML file.

    Returns:
      List[Tuple[int, int, int, int]]: List of bounding boxes in (xmin, ymin, xmax, ymax) format.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []

    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        if bbox is not None:
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # Append the bounding box as a tuple
            boxes.append([xmin, ymin, xmax, ymax])

    return boxes

if __name__ == "__main__":
    app()
