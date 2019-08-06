import os

base_path = os.getcwd()
coco_dataset_year = '2017'
coco_data_path = os.path.join(base_path, "..", "data", "")
coco_caption_path_train = os.path.join(coco_data_path, "annotations", "captions_train{0}.json".format(coco_dataset_year))
coco_caption_path_validation = os.path.join(coco_data_path, "annotations", "captions_val{0}.json".format(coco_dataset_year))
coco_train_image_path = os.path.join(coco_data_path, "re_train{0}".format(coco_dataset_year), "")
coco_validation_image_path = os.path.join(coco_data_path, "re_val{0}".format(coco_dataset_year), "")
coco_test_image_path = os.path.join(coco_data_path, "re_train{0}".format(coco_dataset_year), "")
coco_test_image_path_original = os.path.join(coco_data_path, "train{0}".format(coco_dataset_year), "")
coco_pretrain_image_path = os.path.join(coco_data_path, "re_train{0}".format(coco_dataset_year), "")
coco_pretrain_image_path_original = os.path.join(coco_data_path, "train{0}".format(coco_dataset_year), "")

image_with_caption = os.path.join(base_path, "result", "image_with_caption", "")

heatmap_path = os.path.join(base_path, "result", "heatmap")
heatmap_cnn_path = os.path.join(heatmap_path, "cnn")

train_early_stop = os.path.join(base_path, "result", "early_stop.bin")
model_save_path = os.path.join(base_path, "result", "model", "")
