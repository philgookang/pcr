import argparse
import os
from PIL import Image
from tqdm import tqdm

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))

if __name__ == '__main__':

    image_size = 256
    # target_folders = ["../data/val2017/"] #, "../data/test2017/", "../data/train2017/"
    # result_folders = ["../data/re_val2017/"] #, "../data/re_test2017/", "../data/re_train2017/"
    target_folders = ["../data/flickr8k/"] #, "../data/test2017/", "../data/train2017/"
    result_folders = ["../data/re_flickr8k/"] #, "../data/re_test2017/", "../data/re_train2017/"
    target_folders = ["../data/flickr30k/"] #, "../data/test2017/", "../data/train2017/"
    result_folders = ["../data/re_flickr30k/"] #, "../data/re_test2017/", "../data/re_train2017/"

    for target, result in tqdm(zip(target_folders, result_folders)):
        resize_images(target, result, [image_size, image_size])
