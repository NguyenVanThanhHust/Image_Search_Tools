import os
import logging
import warnings
warnings.filterwarnings('ignore')
import argparse

import flask
from flask import request, jsonify
from PIL import Image

import torch
from torchvision import transforms
from torchvision import datasets

from conf import settings
from utils_ai import build_network

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

def predict_single_img(model, image):
    device = torch.device("cuda")
    image_transforms =  transforms.Compose([
                        transforms.Resize((112, 112)),
                        transforms.ToTensor(),
                        transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)])

    test_img = Image.open(image).convert('RGB')
    test_img_tensor = image_transforms(test_img)

    if torch.cuda.is_available():
        test_img_tensor = test_img_tensor.view(1, 3, 112, 112).cuda()
    else:
        test_img_tensor = test_img_tensor.view(1, 3, 112, 112)
    with torch.no_grad():
        model.eval()
        # model ouputs log probabilities
        out = model(test_img_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(3, dim=1)
        sum_topk = int(topk.cpu().numpy()[0][0]) + int(topk.cpu().numpy()[0][1]) + int(topk.cpu().numpy()[0][2])
    return idx_to_class[topclass.cpu().numpy()[0][0]], (topk.cpu().numpy()[0][0])/sum_topk
    
    
if __name__ == '__main__':
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default= 'squeezenet', help='net type')
    parser.add_argument('-weights', type=str, 
                        default='./checkpoint/results/squeezenet-291-best.pth',
                        help='the weights file path you want to test')
    
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    # keep for training new author
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    
    args_dict = vars(parser.parse_args())
    logger.info(args_dict)

    net_type = args_dict['net']
    use_gpu = args_dict['gpu']
    net = build_network(archi = net_type, use_gpu=use_gpu ) 
    logger.info(net)

    net.load_state_dict(torch.load(args_dict['weights']), args_dict['gpu'])
    net.eval()

    test_image_dir = '../datasets/Image_Search_Dataset/val'
    dataset = datasets.ImageFolder(test_image_dir, transform= None)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    count_authors = len([name for name in os.listdir(test_image_dir) if os.path.isdir(os.path.join(test_image_dir, name))])
    
    original_path = "../datasets/Image_Search_Dataset/val"
    list_classes = next(os.walk(original_path))[1]
    for each_class in list_classes:
        class_path = os.path.join(test_image_dir, each_class)

        for img in os.listdir(class_path):
            test_img_path = os.path.join(class_path, img)
            label, confidence = predict_single_img(net, test_img_path)
            print("Class: ", each_class, ", Predict: ",label, ", Confidence: ", confidence)
            
