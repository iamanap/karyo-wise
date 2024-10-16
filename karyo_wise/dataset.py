from pathlib import Path
from typing import List, Tuple
import PIL.Image
from labelme import utils

import typer
from loguru import logger
from tqdm import tqdm
import xml.etree.ElementTree as ET
import shutil
import csv
import json
import os
import base64

import karyo_wise.config as config

app = typer.Typer()

@app.command()
def main(
  # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
  autokary_dir: Path = config.RAW_AUTOKARY_DIR,
  # ----------------------------------------------
):
  # ---- REPLACE THIS WITH YOUR OWN CODE ----
  logger.info("Processing dataset...")
  generate_interim_data(autokary_dir)
  generate_processed_data()
  logger.success("Processing dataset complete.")
  # -----------------------------------------

def generate_interim_data(autokary_dir: str):
  """
  1. Realocate raw dataset from AutoKary folder to interim img and ann folders
  2. Remove wrong shape labels '-1' and '119' from annotations
  4. Generate csv
  
  """
  logger.info("Generating interim data...")
  autokary_train_dir = autokary_dir / "train_labelme"
  autokary_test_dir = autokary_dir / "test_labelme"

  copy_img_ann_files( # copy train files
    autokary_train_dir,
    config.INTERIM_DATA_DIR_TRAIN_IMG,
    config.INTERIM_DATA_DIR_TRAIN_ANN
  )
  
  copy_img_ann_files( # copy test files
    autokary_test_dir, 
    config.INTERIM_DATA_DIR_TEST_IMG, 
    config.INTERIM_DATA_DIR_TEST_ANN
  )

  json_seg_data_to_csv()

  logger.success("Interim data complete.")

def generate_processed_data():
  """
  Convert json to mask and save to processes masks folder
  """
  logger.info("Generating processes data...")
  train_ann_dir = config.INTERIM_DATA_DIR_TRAIN_ANN
  test_ann_dir = config.INTERIM_DATA_DIR_TEST_ANN

  train_processed_img_dir = config.PROCESSED_DATA_DIR_TRAIN_IMG
  train_processed_masks_dir = config.PROCESSED_DATA_DIR_TRAIN_MASKS
  test_processed_img_dir = config.PROCESSED_DATA_DIR_TEST_IMG
  test_processed_masks_dir = config.PROCESSED_DATA_DIR_TEST_MASKS

  create_dir(train_processed_img_dir)
  create_dir(train_processed_masks_dir)
  create_dir(test_processed_img_dir)
  create_dir(test_processed_masks_dir)

  convert_all_anns_to_img_masks(train_ann_dir, train_processed_img_dir, train_processed_masks_dir)
  convert_all_anns_to_img_masks(test_ann_dir, test_processed_img_dir, test_processed_masks_dir)

  logger.success("Processed data generated")  

def convert_all_anns_to_img_masks(origin_ann_dir_path: str, processed_img_dir: str, processed_mask_dir: str):
  origin_ann_dir_path = Path(origin_ann_dir_path)
  for ann in tqdm(list(origin_ann_dir_path.iterdir()), desc="Converting masks from json to img"):
    labelme_json_shapes_to_label(ann, processed_img_dir, processed_mask_dir)

def labelme_json_shapes_to_label(json_path: str, out_img_dir_path: str, out_label_dir_path: str):
  json_path = Path(json_path)
  out_img_dir = Path(out_img_dir_path)
  out_label_dir = Path(out_label_dir_path)

  data = json.load(json_path.open())

  img_data = data.get("imageData")
  img_path = data["imagePath"]

  # Load JSON data
  if not img_data:
    image_path = json_path.parent / img_path
    with open(image_path, "rb") as f:
      img_data = f.read()
      img_data = base64.b64encode(img_data).decode("utf-8")
  img_arr = utils.img_b64_to_arr(img_data)
  
  # Create label mapping
  label_name_to_value = {}
  for shape in sorted(data["shapes"], key=lambda x: x["label"]):
    label_name = shape["label"]
    if label_name in label_name_to_value:
      label_value = label_name_to_value[label_name]
    else:
      label_value = len(label_name_to_value)
      label_name_to_value[label_name] = label_value
  
  # Create mask
  mask, _ = utils.shapes_to_label(img_arr.shape, data["shapes"], label_name_to_value)

  # Save image and mask
  PIL.Image.fromarray(img_arr).save(out_img_dir / f"{img_path[:-4]}_img.png")
  utils.lblsave(out_label_dir / f"{img_path[:-4]}_mask.png", mask)

def copy_img_ann_files(from_path: str, to_img_path: str, to_ann_path: str):
  create_dir(to_img_path)
  create_dir(to_ann_path)
  
  copy_from_autokary_to_raw(
    from_path, 
    to_img_path, 
    to_ann_path
  )

def create_dir(dir_path: str):
  if os.path.exists(dir_path):
    shutil.rmtree(dir_path)
  os.makedirs(dir_path, exist_ok=False)

def copy_from_autokary_to_raw(autokary_path: str, img_dir_path_to_save: str, ann_dir_path_to_save: str):
  for dir in tqdm(list(autokary_path.iterdir()), desc = "Copying files"):
    if dir.is_dir():
      for file in dir.iterdir():
        if file.suffix == '.png':
          corresponding_annotation_file = Path(file.absolute().with_suffix('.json'))
          if corresponding_annotation_file.exists():
            shutil.copy(file, img_dir_path_to_save)
            shutil.copy(corresponding_annotation_file, ann_dir_path_to_save)
            # Remove "-1" and "119" labels
            remove_useless_labels_from_segments_annotation_file(ann_dir_path_to_save / corresponding_annotation_file.name)
          else:
            logger.debug(f'No corresponding annotation file found for {file}')

def remove_useless_labels_from_segments_annotation_file(json_path: str):
  if not os.path.exists(json_path):
    return
  ori_json = open(json_path)
  data = json.load(ori_json)
  shapes = data["shapes"]
  new_shapes = []

  for shape in shapes:
    label = shape["label"]
    if label != '-1' and label != '119':
      new_shapes.append(shape)
    else:
      logger.debug(f"Path {json_path} has -1 or 119 label")
  data["shapes"] = new_shapes

  with open(json_path, 'w') as file:
    json.dump(data, file, indent=4)

def json_seg_data_to_csv():
  logger.info("Parsing data to csv...")
  parse_dataset(config.INTERIM_DATA_DIR_TRAIN_ANN, config.TRAIN_DATA_CSV, "Train")
  parse_dataset(config.INTERIM_DATA_DIR_TEST_ANN, config.TEST_DATA_CSV, "Test")
  
  logger.success("Data parsed to csv!")
  
def parse_dataset(dataset_path: str, csv_path: str, dataset_type:str):
  dataset = [['ImagePath', 'Label']]
  
  dataset += parse_json_seg_data(dataset_path)
  logger.success(f"{dataset_type} data parsed to csv.")

  with open(csv_path, mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(dataset)
  

def parse_json_seg_data(data_path: str) -> List[List[str]]:
  dataset = []
  data_dir = Path(data_path)
  
  for file in tqdm(list(data_dir.iterdir()), desc="Parsing JSON files"):
    if file.suffix == ".json":
      data = json.load(open(file))
      image_path = data["imagePath"]
      shapes = data["shapes"]

      for shape in shapes:
        label = shape["label"]
        dataset.append([image_path, label])

  return dataset

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
