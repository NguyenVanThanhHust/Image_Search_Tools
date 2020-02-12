import re
import os
import random
import shutil

def get_class_name_from_string(image_path):
    """
    parse string to get class name
    """
    spe_char_1 = '/'
    spe_char_2 = '\\'
    spe_char_3 = '_'
    image_path = str(image_path)
    char_1_pos= [pos for pos, char in enumerate(image_path) if char == spe_char_1]
    char_2_pos= [pos for pos, char in enumerate(image_path) if char == spe_char_2]
    last_spe_char_pos = char_1_pos[-1] if char_1_pos[-1] > char_2_pos[-1] else char_2_pos[-1]
    image_name = image_path[last_spe_char_pos + 1:len(image_path)]
    char_3_pos= [pos for pos, char in enumerate(image_name) if char == spe_char_3]
    class_name = image_name[0:char_3_pos[0]]
    return class_name

def copy_folder(class_name, ori_path, dst_path):
    train_path = os.path.join(dst_path, "train", class_name)
    val_path = os.path.join(dst_path, "val", class_name)
    test_path = os.path.join(dst_path, "test", class_name)
    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    if not os.path.isdir(val_path):
        os.mkdir(val_path)
    if not os.path.isdir(test_path):
        os.mkdir(test_path)

    list_file = next(os.walk(ori_path))[2]
    train_files = random.sample(list_file, int(0.7 * len(list_file)))
    val_test_files = [item for item in list_file if item not in train_files]
    val_files = random.sample(list_file, int(0.5 * len(val_test_files)))
    test_files = [item for item in val_test_files if item not in val_files]

    # train/val/test/ = 0.7/0.15/0.15   
    print("copy: ", len(list_file), " files from: ", ori_path, " to: ", dst_path)
    for each_file in train_files:
        src_file_path = os.path.join(ori_path, each_file)
        dst_file_path = os.path.join(train_path, each_file)
        shutil.copyfile(src_file_path, dst_file_path)
    for each_file in val_files:
        src_file_path = os.path.join(ori_path, each_file)
        dst_file_path = os.path.join(val_path, each_file)
        shutil.copyfile(src_file_path, dst_file_path)
    for each_file in test_files:
        src_file_path = os.path.join(ori_path, each_file)
        dst_file_path = os.path.join(test_path, each_file)
        shutil.copyfile(src_file_path, dst_file_path)    