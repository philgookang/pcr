from config.file import model_file
from config.file import dataset_file

from config.nltk import nltk_to_pos

from config.train import validation_and_test_dataset_size
from config.train import threshold
from config.train import image_crop_size
from config.train import number_of_workers
from config.train import pretrain_number_epochs
from config.train import pretrain_batch_size

from config.coco import base_path
from config.coco import coco_data_path
from config.coco import coco_caption_path_train
from config.coco import coco_image_path ### easier later on
from config.coco import coco_pretrain_image_path
