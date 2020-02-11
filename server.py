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
from shutil import copy

import torch
from torchvision import transforms
from torchvision import datasets

from lshash import LSHash
from conf import settings
from utils_ai import build_network, get_feature_single_img, create_feature
from utils import get_class_name_from_string

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

UPLOAD_FOLDER = './static/uploads'
RESULT_FOLDER = './static/results'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = 'random secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_similar_item_image(imgpath, n_items=3):
    global net
    global feature_dict
    global lsh
    feature = get_feature_single_img(net, imgpath)
    print("image name: ", imgpath)
    response = lsh.query(feature, 
                     num_results=n_items+1, distance_func='l1norm')
    path_1, path_2, path_3 = response[1][0][1], response[2][0][1], response[3][0][1]
    label_1 = get_class_name_from_string(path_1)
    label_2 = get_class_name_from_string(path_2)
    label_3 = get_class_name_from_string(path_3)
    paths = [path_1, path_2, path_3]
    labels = [label_1, label_2, label_3]
    return labels, paths

def allowed_file(filename):
        return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    <title>Objects Classification</title>
    <h1>Objects Classification</h1>
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
 
@app.route('/predict')
def predict():
    global net
    session_keys = list(session.keys())
    # remove old file from previous prediction
    list_file = [f for f in os.listdir(RESULT_FOLDER) if os.path.isfile(os.path.join(RESULT_FOLDER, f))]
    for file in list_file:
        os.remove(os.path.join(RESULT_FOLDER, file))
        
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
        
    labels, paths = get_similar_item_image(imgpath)
    for path in paths:
        copy(path, RESULT_FOLDER)
    list_file = [f for f in os.listdir(RESULT_FOLDER) if os.path.isfile(os.path.join(RESULT_FOLDER, f))]
    filepaths = [os.path.join(RESULT_FOLDER, file) for file in list_file]
    path_1, path_2, path_3 = filepaths[0], filepaths[1], filepaths[2] 
    return render_template('predict.html', 
                           img_1=path_1, img_2 = path_2, img_3 = path_3, 
                           labels_1 = labels[0], labels_2 = labels[1], labels_3 = labels[2])

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
    
    app.run()