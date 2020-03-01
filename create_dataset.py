import logging
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import sys
import random

from utils_common import copy_folder

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.CRITICAL)

logging.info('Start program')
handler = logging.FileHandler('infor.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

def create_data_set(input_dataset_path, output_dataset_path):
    # Randomly choose 10 classes from original datasets:
    classes = next(os.walk(input_dataset_path))[1]
    list_using_class = random.sample(classes, 10)
    train_path = os.path.join(output_dataset_path, "train")
    val_path = os.path.join(output_dataset_path, "val")
    test_path = os.path.join(output_dataset_path, "test")
    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    if not os.path.isdir(val_path):
        os.mkdir(val_path)
    if not os.path.isdir(test_path):
        os.mkdir(test_path)
        
    for each_class in list_using_class:
        ori_path = os.path.join(input_dataset_path, each_class)
        dst_path = output_dataset_path
        copy_folder(each_class, ori_path, output_dataset_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-idp', type=str, default= '../../datasets/101_ObjectCategories', help='input dataset path')
    parser.add_argument('-odp', type=str, default= '../../datasets/Image_Search_Dataset', help='output dataset path')

    args_dict = vars(parser.parse_args())
    logger.info(args_dict)
    input_dataset_path = args_dict['idp']
    output_dataset_path = args_dict['odp']
    if not os.path.isdir(input_dataset_path):
        print("This is not a folder")
        sys.exit()
    if not os.path.isdir(output_dataset_path):
        os.mkdir(output_dataset_path)
    create_data_set(input_dataset_path, output_dataset_path)
