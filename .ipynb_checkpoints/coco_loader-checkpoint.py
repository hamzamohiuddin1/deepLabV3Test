import os
from glob import glob
import tensorflow as tf
from PIL import Image

COCO_PATH = '/ocean/datasets/community/COCO/Dataset_2017'

def json_to_mask(json_path, mask_save_dir):
    # Function to convert COCO JSON annotations to segmentation masks
    # You may need to install the required libraries, e.g., via: pip install pycocotools
    from pycocotools.coco import COCO
    import numpy as np

    coco = COCO(json_path)
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Create an empty mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        for ann in anns:
            seg = ann['segmentation']
            category_id = ann['category_id']

            # Draw segmentation on the mask
            mask = coco.annToMask(ann) * category_id
            
        mask_name = f"{os.path.splitext(img_info['file_name'])[0]}_mask.png"
        mask_path = os.path.join(mask_save_dir, mask_name)
        # Save the mask
        Image.fromarray(mask).save(mask_path)

if __name__ == "__main__":
    # Specify paths
    coco_train_json = os.path.join(COCO_PATH, 'annotations','instances_train2017.json')
    coco_val_json = os.path.join(COCO_PATH, 'annotations','instances_val2017.json')
    coco_mask_save_dir_train = './DeepLabV3Plus/dataset/coco_masks/trainannot'
    coco_mask_save_dir_val = './DeepLabV3Plus/dataset/coco_masks/valannot'

    # Create directories if they don't exist
    os.makedirs(coco_mask_save_dir_train, exist_ok=True)
    os.makedirs(coco_mask_save_dir_val, exist_ok=True)
    
    # Convert COCO JSON annotations to segmentation masks
    json_to_mask(coco_train_json, coco_mask_save_dir_train)
    json_to_mask(coco_val_json, coco_mask_save_dir_val)

# Make sure to replace './path/to/Coco/' with the actual path to your COCO dataset.
