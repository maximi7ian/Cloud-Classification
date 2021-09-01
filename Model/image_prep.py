import boto3
import numpy as np
import os
from tqdm import tqdm
import random
from PIL import Image

s3 = boto3.client('s3')

def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()

    for entry in listOfFile:

        fullPath = os.path.join(dirName, entry)

        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


def lbl_switch(arg, lbl2int = True):

    switcher = {
            'BlueSky': 0,
            'Patterned': 1,
            'ThickDark': 2,
            'ThickWhite': 3,
            'Veil': 4
        }

    if lbl2int:
        return switcher.get(arg)
        
    else:
        return list(switcher.keys())[list(switcher.values()).index(arg)]



if __name__ == '__main__':

    bucket_name = 'BUCKET NAME'
    train_x_path = 'TRAIN X PATH'
    train_y_path = 'TRAIN Y PATH'
    test_x_path = 'TEST X PATH'
    test_y_path = 'TEST Y PATH'
    
    fls = getListOfFiles('./Model/Data')
    random.shuffle(fls)

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for img in tqdm(fls):

        lbl = img.split('/')[2]
        nme = img.split('_')[1]
        key_name = lbl + '_' + nme

        lbl = lbl_switch(lbl)

        if 'png' in img:

            if np.random.uniform() > 0.2:
                np_im = np.asarray(Image.open(img))
                x_train.append(np_im)
                y_train.append(lbl)
            else:
                np_im = np.asarray(Image.open(img))
                x_test.append(np_im)
                y_test.append(lbl)

    x_train = np.stack(x_train, axis=0)
    x_test = np.stack(x_test, axis=0)


    np.save('./train_data.npy', np.array(x_train))
    np.save('./train_labels.npy', np.array(y_train))
    np.save('./test_data.npy', np.array(x_test))
    np.save('./test_labels.npy', np.array(y_test))


    s3.upload_file('./train_data.npy', bucket_name, train_x_path)
    s3.upload_file('./train_labels.npy', bucket_name, train_y_path)

    s3.upload_file('./test_data.npy', bucket_name, test_x_path)
    s3.upload_file('./test_labels.npy', bucket_name, test_y_path)