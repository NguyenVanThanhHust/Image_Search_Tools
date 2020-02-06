#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
#from dataset import *

#from skimage import io
from matplotlib import pyplot as plt
from torchvision import datasets
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from conf import settings
from utils import get_network, get_test_dataloader

from PIL import Image
from torch import unsqueeze
import time
### python test.py -net squeezenet -weights /home/doannn/Documents/Working_Server/test/pytorch-cifar100/checkpoint/squeezenet/results_ckp/squeezenet-121-best.pth

### Predict images
def predict_image():
    image_transforms = { 
        'test': transforms.Compose([
            transforms.Resize(size=112),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    }
    print ("2222")
    transform = image_transforms['test']
    test_img = Image.open('/home/doannn/Documents/Working_Server/datasets/SealDataset_TrainValTest/test/0018_KawaiGyokudo/611-0375_0.jpg') 
    test_img_tensor = transform(test_img)

    test_image_path = 'D:/SealProject/Datasets/images/val'
    test_dataset = datasets.ImageFolder(test_image_path, transform= None)

    # if torch.cuda.is_available():
    # test_img_tensor = test_img_tensor.view(112, 112).cuda()
    # else:
    #     test_img_tensor = test_img_tensor.view(1, 3, 112, 112)

    with torch.no_grad():
        net.eval()
        # model outputs probabilities
        out = net(test_dataset)
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim= 1)
        print("Output class :  ", test_dataset.idx_to_class[topclass.cpu().numpy()[0][0]])


    
if __name__ == '__main__':
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default= 'squeezenet', help='net type')
    ###21 & 51  
    parser.add_argument('-weights', type=str, default='./checkpoint/squeezenet/checkpoint_results/squeezenet-151-best.pth', help='the weights file path you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    args = parser.parse_args()

    net = get_network(args) 

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        # settings.CIFAR100_PATH,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    net.load_state_dict(torch.load(args.weights), args.gpu)
    # print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    for n_iter, (image, label) in enumerate(cifar100_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        output = net(image)
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        label = label.view(label.size(0), -1).expand_as(pred)
        # print ("label: %s"%label)
        correct = pred.eq(label).float()
        # print (correct)

        #compute top 5
        correct_5 += correct[:, :5].sum()

        #compute top1 
        correct_1 += correct[:, :1].sum()
    
    # print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 1 correct: ", correct_1 / len(cifar100_test_loader.dataset))
    # print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Top 5 correct: ", correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))


    # ### Predict images
    # transform_test = transforms.Compose([
    #         # transforms.ToPILImage(),
    #         transforms.Resize((112, 112)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
    # ])
    # test_image = Image.open('/home/doannn/Documents/Working_Server/datasets/SealDataset_TrainValTest/val/1_hirayama/1_hirayama_original_239-0173_0.jpg_c6187934-635c-4928-b5df-cba044635432.jpg')
    # # test_image = Variable(test_image).cuda()
    # x = transform_test(test_image) # preprocess img
    # x = x.unsqueeze(0) # add batch dimension
    
    # output = net(x)
    # pred = torch.argmax(output, 1)
    # print('Image predicted as ', pred)

## Predict images
    # transform_test = transforms.Compose([
    #         # transforms.ToPILImage(),
    #         transforms.Resize((112, 112)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
    # ])
    # test_image = Image.open('/home/doannn/Documents/Working_Server/datasets/SealDataset_TrainValTest/val/1_hirayama/1_hirayama_original_239-0173_0.jpg_c6187934-635c-4928-b5df-cba044635432.jpg')
    # # test_image = Variable(test_image).cuda()
    # x = transform_test(test_image) # preprocess img
    # x = x.unsqueeze(0) # add batch dimension
    
    # output = net(x)
    # pred = torch.argmax(output, 1)
    # print('Image predicted as ', pred)



    ### Predict a folder (images ..)
    image_transforms = { 
        'test': transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)])
    }

    testset = {'predict': datasets.ImageFolder('/home/doannn/Documents/Working_Server/datasets/SealDataset_TrainValTest/test', 
                image_transforms['test'])}
    testloader = {'predict': torch.utils.data.DataLoader(testset['predict'], shuffle=False, num_workers=4, batch_size=16)}

    # outputs = list()
    output = list()
    since = time.time()
    for image, label in testloader['predict']: ## image, label <-> torch.Tensor
        image = image.to(device)
        output = net(image)
        output = output.to(device)
        # print (type(output))
        index = output.cpu().data.numpy().argmax() #data is first moved to cpu and then converted to numpy array
        # print (type(index))

    ## Functions
    predict_image()
