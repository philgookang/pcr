import os
from config.config_train import dataset_type

BASE_PATH = os.getcwd()
COCO_ROOT = os.path.join(BASE_PATH, "..", "data", "")
RESULT_ROOT = os.path.join(BASE_PATH, "result")

COCO_IMAGE_PATH = os.path.join(COCO_ROOT, "coco2017", "")
COCO_TRAIN_ANNOTATION = os.path.join(COCO_ROOT, "annotations", "captions_train2017.json")
COCO_VALIDATION_ANNOTATION = os.path.join(COCO_ROOT, "annotations", "captions_val2017.json")

FLICKR8k_IMAGE_PATH = os.path.join(COCO_ROOT, "re_flickr8k", "")
FLICKR8k_TRAIN_IMG = os.path.join(COCO_ROOT, "Flickr_8k.trainImages.txt")
FLICKR8k_VALIDATION_IMG = os.path.join(COCO_ROOT, "Flickr_8k.devImages.txt")
FLICKR8k_TEST_IMG = os.path.join(COCO_ROOT, "Flickr_8k.testImages.txt")
FLICKR8k_ANNOTATION = os.path.join(COCO_ROOT, "Flickr8k.token.txt")

FLICKR30k_IMAGE_PATH = os.path.join(COCO_ROOT, "re_flickr30k", "")
FLICKR30k_ANNOTATION = os.path.join(COCO_ROOT, "flickr30k.txt")

RESULT_IMAGE_W_CAPTION = os.path.join(RESULT_ROOT, "image_with_caption", "")
RESULT_HEATMAP_ROOT = os.path.join(RESULT_ROOT, "heatmap")
RESULT_HEATMAP_CNN_PATH = os.path.join(RESULT_HEATMAP_ROOT, "cnn")

RESULT_MODEL_PATH = os.path.join(RESULT_ROOT, "model", "")
RESULT_DATASET_PATH = os.path.join(RESULT_ROOT, "dataset", "")

IMG_PATH = COCO_IMAGE_PATH # default 1
if dataset_type == 2:
    IMG_PATH = FLICKR8k_IMAGE_PATH
elif dataset_type == 3:
    IMG_PATH = FLICKR30k_IMAGE_PATH
