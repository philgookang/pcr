import os

BASE_PATH = os.getcwd()
COCO_ROOT = os.path.join(BASE_PATH, "..", "data", "")
RESULT_ROOT = os.path.join(BASE_PATH, "result")

COCO_IMAGE_PATH = os.path.join(COCO_ROOT, "coco2017", "")
COCO_TRAIN_ANNOTATION = os.path.join(COCO_ROOT, "annotations", "captions_train2017.json")
COCO_VALIDATION_ANNOTATION = os.path.join(COCO_ROOT, "annotations", "captions_val2017.json")

RESULT_IMAGE_W_CAPTION = os.path.join(RESULT_ROOT, "image_with_caption", "")
RESULT_HEATMAP_ROOT = os.path.join(RESULT_ROOT, "heatmap")
RESULT_HEATMAP_CNN_PATH = os.path.join(RESULT_HEATMAP_ROOT, "cnn")

RESULT_MODEL_PATH = os.path.join(RESULT_ROOT, "model", "")
RESULT_DATASET_PATH = os.path.join(RESULT_ROOT, "dataset", "")
