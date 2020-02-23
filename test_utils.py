from utils_common import get_class_name_from_string

if __name__ == '__main__':
    image_path = '../datasets/Image_Search_Dataset/train/buddha/image_0036.jpg'
    name_img = get_class_name_from_string(image_path)
    print(name_img)
    