from utils_common import get_class_name_from_string

if __name__ == '__main__':
    image_path = '../datasets/Signature_Recognition/To_Use_Datasets/val/GotoSumio/GotoSumio_1_423-0666.jpg'
    name_img = get_class_name_from_string(image_path)
    print(name_img)
    