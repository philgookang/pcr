import os

base_path = os.getcwd()
coco_data_path = os.path.join(base_path, "..", "data", "")
coco_caption_path_train = os.path.join(coco_data_path, "annotations", "captions_train2017.json")
coco_caption_path_validation = os.path.join(coco_data_path, "annotations", "captions_val2017.json")
coco_image_path = os.path.join(coco_data_path, "resized2014", "")
coco_pretrain_image_path = os.path.join(coco_data_path, "re_train2017", "")
