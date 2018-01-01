#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: shai ehrlich
"""

import cPickle
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
import time
from sklearn.feature_extraction import image as fe_image
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os.path
import sys
import getopt

data_path = './'

rand = np.random.RandomState(0)

""" Parameters of feature extraction and unsupervised learning """
# receptive field size
w = 6
# number of channels (RGB)
d = 3
patch_size = (w, w)
patch_shape = (w, w, d)
step_size = 1
patch_len = w * w * d
features_width = 32 - w + step_size
features_len = features_width*features_width
max_patches_per_img = 40
n_clusters = 1600

# globals
kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=n_clusters, 
                         random_state=rand)
svm = LinearSVC(dual=False)
clusters = []
ZPWMatrix = []


def train(path=None):
    """ Train classifier. If path is not None, will save the K-means clusterd 
    and SVM model to the given path. """
    global clusters
    start = time.time()
    
    # Feature learning using k-means
    filename = os.path.join(data_path, 'data_batch_4')
    trainingSetAImages, trainingSetALabels = create_images_set(filename)
    patches = feature_learning_extract_patches(trainingSetAImages)
    print 'starting K-means'
    feature_learning_KMeans(patches)

    # Feature extraction and classification
    filename = os.path.join(data_path, 'data_batch_2')
    trainingSetBImages, trainingSetBLabels = create_images_set(filename)
    trainingFeatures = extract_features(trainingSetBImages)
    train_classifier(trainingFeatures, trainingSetBLabels)
    
    end = time.time()
    print 'total training time: ', str(int((end - start)/60.0)), 'min'
    
    if path is not None:
        model = {}
        model['clusters'] = clusters
        model['svm'] = svm
        with open(os.path.join(path,'model'), 'wb') as handle:
            cPickle.dump(model, handle, protocol=cPickle.HIGHEST_PROTOCOL)
    
def verify(path=''):
    """ Verify the trained model. If path is not None, will load the K-means 
    clusters and SVM model from the given path. """
    global clusters
    
    if path is not None:
        load_model(path)
        
    # Validation
    filename = os.path.join(data_path, 'test_batch')
    validationImages, vaidationLabels = create_images_set(filename)
    validationFeatures = extract_features(validationImages)
    prediction = predict(validationFeatures)
    
    score = accuracy_score(prediction, vaidationLabels)
    print 'accuracy score: ', score
    print 'error rate: ', 1.0-score
        
    show_confusion_matrix(vaidationLabels, prediction)

def create_images_set(filename):
    """ Load image set from filename """
    labels = []
    images = []
    dict = unpickle(filename)
    for idx in range(len(dict["labels"])):
        label = dict["labels"][idx]
        img = dict["data"][idx].reshape(3,32,32).transpose(1,2,0).astype(np.float32)
        labels.append(label)
        images.append(img)
    return images, labels
    
def feature_learning_extract_patches(images):
    """ Extract random patches from images, normalize and whiten """
    global ZPWMatrix
    start = time.time()
    all_patches = []
    for img in images:
        # extract random patches from image
        patches = fe_image.extract_patches_2d(img, patch_size, 
                                            max_patches=max_patches_per_img, 
                                            random_state=rand)
        # reshape to 2D matrix
        patches = np.reshape(patches, (max_patches_per_img, patch_len))
        # normalization
        patches = scale(patches, axis=1)
        if len(all_patches) == 0:
            all_patches = patches
        else:
            all_patches = np.vstack((all_patches, patches))
    # Zero phase whitening
    ZPWMatrix = zero_phase_whitening(all_patches.T)
    all_patches = np.dot(ZPWMatrix, all_patches.T)
    end = time.time()
    print 'feature_learning_extract_patches: ', str(int((end - start)/60.0)), 'min'
    return all_patches.T.astype(np.float32)

def feature_learning_KMeans(patches):
    """ Perform K-means on patches to find cluster centers """
    global clusters
    start = time.time()
    kmeans.fit(patches)
    end = time.time()
    print 'feature_learning_KMeans: ', str(int((end - start)/60.0)), 'min'
    clusters = kmeans.cluster_centers_

def zero_phase_whitening(X):
    """ Whiten patches """
    sigma = np.cov(X, rowvar=True)
    U,S,V = np.linalg.svd(sigma)
    epsilon = 1e-5
    ZPWMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
    return ZPWMatrix

def compute_feature_mapping(patch):
    """ Compute feature mapping for a patch """
    # flatten to a vector
    p = patch.flatten(order='F').astype(np.float32)
    # replicate patch vector the number of clusters
    P = np.tile(p, (n_clusters, 1))
    # compute the norm of the patch vector from each cluster center
    Z = np.linalg.norm(P-clusters, axis=1)   
    Z_mean = np.mean(Z)
    f = np.maximum(0, Z_mean-Z)
    return f

def extract_features_from_image(image):
    """ Extract features from image using a w-by-w window and a step size """
    patches = fe_image.extract_patches(image, 
                                       patch_shape=patch_shape, 
                                       extraction_step=step_size)

    patches = patches.reshape(-1, w, w, d)
    features = []
    # compute mapping function for each patch and append to list
    for patch in patches:
        feature = compute_feature_mapping(patch)
        features.append(feature)
    features = np.asarray(features)
    # reshape
    features = features.reshape(features_width, features_width, n_clusters)
    return features.astype(np.float32)

def extract_features_pooling(features):
    """ Pool the extracted features to create a single vector of size 
        4*n_clusters """
    """ ----------- 
        |sum0|sum1|
        -----------
        |sum2|sum3|
        -----------
    """
    # split each channel to quadrants and sum each quad
    # 1. split along the rows
    split_rows = np.array_split(features, 2, axis=0)
    
    # 2. split along columns
    split_col_a = np.array_split(split_rows[0], 2, axis=1)
    split_col_b = np.array_split(split_rows[1], 2, axis=1)
    
    # 3. sum each quadrant
    sum0 = np.sum(split_col_a[0], dtype=np.float32, axis=0).sum(0)
    sum1 = np.sum(split_col_a[1], dtype=np.float32, axis=0).sum(0)
    sum2 = np.sum(split_col_b[0], dtype=np.float32, axis=0).sum(0)
    sum3 = np.sum(split_col_b[1], dtype=np.float32, axis=0).sum(0)
    
    # 4. construct a feature vector
    feature_vector = np.concatenate((sum0, sum1, sum2, sum3))
    return feature_vector

def extract_features(images):
    """ Extract features from all images """
    print 'Starting to extract features from images'
    start = time.time()
    all_features = []
    count = 0
    for img in images:
        count +=1
        # extract features from image
        features = extract_features_from_image(img)
        
        #pooling
        feature_vector = extract_features_pooling(features)
        all_features.append(feature_vector)
        if count % 100 == 0:
            print 'extract features from image #', count
    end = time.time()
    print 'extract_features: ', str(int((end - start)/60.0)), 'min'
    return all_features

def train_classifier(feature_vectors, classes):
    start = time.time()
    svm.fit(feature_vectors, np.array(classes))
    end = time.time()
    print 'train_classifier: ', str(int((end - start)/60.0)), 'min'
    
def predict(feature_vectors):
    """ Predict image classes from image features """
    res = svm.predict(feature_vectors)
    return res

def unpickle(filename):
    with open(filename, mode='rb') as file:
        data = cPickle.load(file)
        return data

    return data

def show_confusion_matrix(vaidationLabels, prediction, normalize=True):
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(vaidationLabels, prediction)
    plt.figure()
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes=unpickle(os.path.join(data_path, 'batches.meta'))["label_names"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    normalize=False
    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def load_model(filename):
    """ Load model from file """
    global clusters
    global svm
    
    if not os.path.exists(filename):
        return
    with open(filename, mode='rb') as file:
        model = cPickle.load(file)

    clusters = model['clusters']
    svm = model['svm']
    return

def main(argv):
    bTrain = False
    bVerify = False
    model_path = ''
    global data_path
    
    try:
        opts, args = getopt.getopt(argv,"htv:d:",["train=","verify=","data="])
    except getopt.GetoptError:
        print 'classifier.py -t|--train -d|--data <path-to-cifar-batches>'
        print 'classifier.py -v|--verify <path-to-model-file> -d|--data <path-to-cifar-batches>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'classifier.py -t|--train -d|--data <path-to-cifar-batches>'
            print 'classifier.py -v|--verify <path-to-model-file> -d|--data <path-to-cifar-batches>'        
            sys.exit()
        elif opt in ("-t", "--train"):
            bTrain = True
        elif opt in ("-v", "--verify"):
            bVerify = True
            model_path = arg
        elif opt in ("-d", "--data"):
            data_path = arg
            
    if bTrain:
        train()
    elif bVerify:
        verify(model_path)
        
    
if __name__ == '__main__':
    main(sys.argv[1:])