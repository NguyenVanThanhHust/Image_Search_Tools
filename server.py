import logging
import warnings
warnings.filterwarnings('ignore')
import argparse
import json
import os
import pickle

import webbrowser
from flask import Flask, request, render_template, flash, redirect, url_for, session, jsonify
from werkzeug import secure_filename
from PIL import Image

import torch
from torchvision import transforms
from torchvision import datasets

from lshash import LSHash
from conf import settings
from utils import build_network

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

def get_similar_item_image(imgpath, n_items=5):
    global net
    global feature_dict
    global lsh
    feature = get_feature_single_img(imgpath)
    print("image name: ", imgpath)
    response = lsh.query(feature, 
                     num_results=n_items+1, distance_func='l1norm')
    return response[0][0][1], response[3][0][1], response[2][0][1]

def get_similar_item(idx, feature_dict, lsh_variable, n_items=5):
    print("image name: ", list(feature_dict.keys())[idx])
    response = lsh_variable.query(feature_dict[list(feature_dict.keys())[idx]].flatten(), 
                     num_results=n_items+1, distance_func='l1norm')
    return response[0][0][1], response[3][0][1], response[2][0][1]

def get_feature_single_img(image_path):
    global net
    """
    
    """
    device = torch.device("cuda")
    image_transforms =  transforms.Compose([
                        transforms.Resize((112, 112)),
                        transforms.ToTensor(),
                        transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, 
                                             settings.CIFAR100_TRAIN_STD)])

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

def predict_author_single_img(model, image_path):
    """
    
    """
    device = torch.device("cuda")
    image_transforms =  transforms.Compose([
                        transforms.Resize((112, 112)),
                        transforms.ToTensor(),
                        transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, 
                                             settings.CIFAR100_TRAIN_STD)])

    img = Image.open(image_path)
    img_tensor = image_transforms(img)

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

        topk, topclass = ps.topk(3, dim=1)
        
        sum_topk = int(topk.cpu().numpy()[0][0]) + int(topk.cpu().numpy()[0][1]) + int(topk.cpu().numpy()[0][2])
    return idx_to_class[topclass.cpu().numpy()[0][0]], (topk.cpu().numpy()[0][0])/sum_topk, feature


def allowed_file(filename):
        return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = 'random secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        url = request.form['url'] if 'url' in request.form else ''
        if url:
            session['url'] = url
            session.pop('imgpath') if 'imgpath' in session.keys() else None
            return redirect(url_for('predict'))
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No seleced file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            imgpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(imgpath)
            #img = Image.open(imgpath).convert('RGB')
            session['imgpath'] = imgpath
            session.pop('url') if 'url' in session.keys() else None
            return redirect(url_for('predict'))
    return '''
    <!DOCTYPE html>
    <title>Cars Classification</title>
    <h1>Cars Classification</h1>
    <h2>Upload new file</h2>
    <form method=post enctype=multipart/form-data>
        Image File:<input type=file name=file>
        <input type=submit value=Upload>
    </form>
    <h2>OR Paste URL:</h2>
    <form method=post>
        URL:<input type=url name=url>
        <input type=submit value=Go>
    </form>
    '''
 
def classify_image(img, imgpath=""):
    img_tensor = preprocess(img).unsqueeze(0)
    features_var = model(Variable(img_tensor)).data.numpy()
    num_input_features = len(features_var)
    k = 5

    _, I = index.search(features_var, k)
    for i in range(num_input_features):
        suggested_labels = []
        suggested_filenames = []
        for idx in I[i]:
            suggested_filenames.append(train_labels[idx])
            suggested_label = train_labels[idx].split('/')[0]
            suggested_labels.append(suggested_label)
        print("suggested_labels", suggested_labels)
    pred_class = suggested_labels[0]

    app.logger.info("Execution time: %0.02f seconds" % (dt))
    app.logger.info("Image %s classified as %s" % (imgpath, pred_class))

    return (suggested_labels, suggested_filenames)
    
@app.route('/predict')
def predict():
    global net
    session_keys = list(session.keys())
    url = session['url'] if 'url' in session_keys else ''
    imgpath = session['imgpath'] if 'imgpath' in session_keys else ''
    if url:
        imgpath = url
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    elif imgpath:
        imgpath = session['imgpath']
        img = Image.open(imgpath).convert('RGB')
    else:
        imgpath = 'static/uploads/rav4.jpg'
        img = Image.open(imgpath).convert('RGB')
    # path_1, path_2, path_3 = get_similar_item_image(img, imgpath=imgpath)
    
    path_1, path_2, path_3 = get_similar_item_image(imgpath)
    print(path_1)
    print(path_2)
    print(path_3)
    # return render_template('predict.html', suggested_labels=suggested_labels, imgpath=imgpath, suggested_filenames=suggested_filenames)
    # return render_template('predict.html', suggested_labels="Haruki Murakimi", imgpath=path_2, suggested_filenames=path_2)
    img_1 = Image.open(path_2)
    img_1.show() 
    return 'ok'

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
            
if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default= 'squeezenet', help='net type')
    parser.add_argument('-weights', type=str, default='./checkpoint/results/sign_squeezenet-280-regular.pth', help='the weights file path you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    
    args_dict = vars(parser.parse_args())
    logger.info(args_dict)

    net_type = args_dict['net']
    use_gpu = args_dict['gpu']
    net = build_network(archi = net_type, use_gpu=use_gpu) 
    # logger.info(net)

    net.load_state_dict(torch.load(args_dict['weights']), args_dict['gpu'])
    net.eval()

    example_image_dir = 'D:/SealProjectOLD/Datasets/images/val'
    dataset = datasets.ImageFolder(example_image_dir, transform= None)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    json_file = open('author_data.json')
    list_author = json.load(json_file)
    
    if os.path.isfile('lsh.p'):
        logger.info("load indexed dict")
        lsh = pickle.load(open('lsh.p','rb'))
        feature_dict = pickle.load(open('feature_dict.p','rb'))
    else:
        logger.info("building dict")
        lsh, image_paths, list_features = create_feature(list_author, net) 
        feature_dict = dict(zip(image_paths, list_features))
        pickle.dump(feature_dict, open("feature_dict.p", "wb"))
    
    for i in range(10):
        path_1, path_2, path_3 = get_similar_item(i, feature_dict, lsh,5)
        print(path_1)
        print(path_2)
        print(path_3)
        print(10 * "---")
        app.run()