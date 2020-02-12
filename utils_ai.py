""" helper function

author baiyu
"""

import sys

import numpy

import torch
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from conf import settings

#from dataset import CIFAR100Train, CIFAR100Test
def build_network(archi = 'squeezenet', use_gpu=True):
    """ return given network
    """

    if archi == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif archi == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif archi == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif archi == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif archi == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif archi == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif archi == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif archi == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif archi == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif archi == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif archi == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif archi == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif archi == 'xception':
        from models.xception import xception
        net = xception()
    elif archi == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif archi == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif archi == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif archi == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif archi == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif archi == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif archi == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif archi == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif archi == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif archi == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif archi == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif archi == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif archi == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif archi == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif archi == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif archi == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif archi == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif archi == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif archi == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif archi == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif archi == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif archi == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif archi == 'seresnet34':
        from models.senet import seresnet34 
        net = seresnet34()
    elif archi == 'seresnet50':
        from models.senet import seresnet50 
        net = seresnet50()
    elif archi == 'seresnet101':
        from models.senet import seresnet101 
        net = seresnet101()
    elif archi == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    if use_gpu:
        net = net.cuda()

    return net

def get_training_dataloader(mean, std, train_path="",batch_size=32, num_workers=4, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    training = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
    training_loader = torch.utils.data.DataLoader(
        training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return training_loader

def get_test_dataloader(mean, std, test_path="", batch_size=32, num_workers=4, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
	    transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_path = 'D:/SealProject/Datasets/images/val'
    test = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)
    test_loader = DataLoader(
        test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    idx_to_class = {v: k for k, v in test.class_to_idx.items()}

    return test_loader, idx_to_class

def compute_mean_std(dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        training_dataset or test_dataset
        witch derived from class torch.utils.data
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([dataset[i][1][:, :, 0] for i in range(len(dataset))])
    data_g = numpy.dstack([dataset[i][1][:, :, 1] for i in range(len(dataset))])
    data_b = numpy.dstack([dataset[i][1][:, :, 2] for i in range(len(dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    
def get_feature_single_img(net, image_path):
    device = torch.device("cuda")
    image_transforms =  transforms.Compose([
                        transforms.Resize((112, 112)),
                        transforms.ToTensor(),
                        transforms.Normalize(settings.TRAIN_MEAN, 
                                             settings.TRAIN_STD)])

    img = Image.open(image_path)
    img_tensor = image_transforms(img)
    model = net
    if torch.cuda.is_available():
        img_tensor = img_tensor.view(1, 3, 112, 112).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 112, 112)
    with torch.no_grad():
        model.eval()
        # model ouputs log probabilities
        out = model(img_tensor)  # <class 'torch.Tensor'>  torch.Size([1, 58])
        ps = torch.exp(out) #  <class 'torch.Tensor'> torch.Size([1, 58])
        feature = ps.cpu().numpy()[0]
    return feature

def create_feature(list_author, net):
    global example_image_dir
    list_feature = list()
    image_paths = list()
    ## Locality Sensitive Hashing
    k = 10 # hash size
    L = 5  # number of tables
    d = 58 # Dimension of Feature vector
    lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)
    for subfolder in list_author.keys():
        subfolder_path = os.path.join(example_image_dir, subfolder)
        count_items = len([name for name in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, name))])
        # print(subfolder)
        sum_acc = 0
        sum_confiden = 0

        for img in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, img)
            author, confidence, feature = predict_author_single_img(net, image_path)
            image_paths.append(image_path)
            list_feature.append(feature)
            lsh.index(feature, extra_data=image_path)
    pickle.dump(lsh, open('lsh.p', "wb"))
    return lsh, image_paths, list_feature