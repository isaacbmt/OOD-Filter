import torchvision
from shutil import copy2
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib
import re
import argparse
import logging
matplotlib.use('Agg')
import os
import random
# import copy
import ntpath
#OOD flag
OOD_LABEL = -1
import numpy as np
import shutil
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from random import randint

import torch


def create_train_test_folder_partitions_simple(datasetpath_base, percentage_evaluation=0.25, random_state=42,
                                               batch=0, create_dirs = True, num_train_samples=20,
                                               num_test_samples=10, num_classes=2):
    """
    Train and test partitioner
    :param datasetpath_base:
    :param percentage_used_labeled_observations: The percentage of the labeled observations to use from the 1 -  percentage_evaluation
    :param num_batches: total number of batches
    :param create_dirs:
    :param percentage_evaluation: test percentage of data
    :return:
    """
    datasetpath_test = datasetpath_base + r"/labeled_batches/batch_" + str(batch) + r"/test/ "[:-1]
    datasetpath_train = datasetpath_base + r"/labeled_batches/batch_" + str(batch) + r"/train/ "[:-1]
    datasetpath_all = datasetpath_base + r"/all"
    print("datasetpath_all")
    print(datasetpath_all)
    dataset = torchvision.datasets.ImageFolder(datasetpath_all)
    list_file_names_and_labels = dataset.imgs
    labels_temp = dataset.targets
    list_file_names = []
    list_labels = []
    # list of file names and labels
    for i in range(0, len(list_file_names_and_labels)):
        file_name_path = list_file_names_and_labels[i][0]
        list_file_names += [file_name_path]
        list_labels += [labels_temp[i]]

    if (create_dirs):
        # create the directories
        print("Trying to create dir")
        print(datasetpath_test)
        os.makedirs(datasetpath_test)
        print(datasetpath_test)
        os.makedirs(datasetpath_train)
        for i in range(0, num_classes):
            os.makedirs(datasetpath_test + "/ "[:-1] + str(i))
            os.makedirs(datasetpath_train + "/ "[:-1] + str(i))

    # test and train  splitter for unlabeled and labeled data split

    X_train, X_test, y_train, y_test = train_test_split(list_file_names, list_labels, test_size=percentage_evaluation,
                                                        random_state=random_state)

    print("Creating trainig partitioned folders...", len(X_train))
    for i in range(0, num_train_samples):
        # print(X_train[i] + " LABEL: " + str(y_train[i]))
        path_src = X_train[i]
        # extract the file name
        file_name = ntpath.basename(path_src)
        # print("File name", file_name)
        path_dest = datasetpath_train + str(y_train[i]) + "/ "[:-1] + file_name
        # print("COPY TO: " + path_dest)
        copy2(path_src, path_dest)

    print("Creating test partitioned folders...", len(X_test))
    for i in range(0, num_test_samples):
        # print(X_test[i] + " LABEL: " + str(y_test[i]))
        path_src = X_test[i]
        file_name = ntpath.basename(path_src)
        # print("File name", file_name)
        path_dest = datasetpath_test + str(y_test[i]) + "/ "[:-1] + file_name
        # print("COPY TO: " + path_dest)
        copy2(path_src, path_dest)


def create_folder_partitions_unlabeled_ood(labeled_dataset_path, iod_dataset_path, ood_dataset_path, dest_unlabeled_path_base,
                                           total_unlabeled_obs=1000, ood_percentage=0.5, random_state=42, batch=0,
                                           create_dirs=True, num_classes=2):
    """
    Create the folder partitions for unlabeled data repository, preserving the folder structure of train data
    This MUST BE EXECUTED AFTER the training batches have been built
    The OOD data is randomly copied among the training subfolders, given the folder structure used in the MixMatch FAST AI implementation
    :param iod_dataset_path:
    :param ood_dataset_path:
    :param dest_unlabeled_path_base: We create the train folder, as the test folder is just copied from IOD folder (test is always In Distribution)
    :param total_unlabeled_obs:
    :param ood_percentage: percentage of out of distribution data
    :param random_state: seed
    :param batch: batch number id for the folder
    :param create_dirs: create the necessary directories
    :return:
    """
    # read the data from the in distribution dataset train batch (the selected observations for unlabeled data will be deleted from there)
    dataset = torchvision.datasets.ImageFolder(iod_dataset_path + "/all")
    print(f"ood path: {ood_dataset_path}")
    dataset_ood = torchvision.datasets.ImageFolder(ood_dataset_path)
    # read the data path
    list_file_names_in_dist_data = dataset.imgs
    list_file_names_out_dist_data = dataset_ood.imgs

    #CORRECT! MUST BE THE FOLDER NAME, AND NOT THE AUTOMATED TARGET ERROR
    in_dist_classes_list_all = os.listdir(iod_dataset_path + "/all")
    print("NEW LABELS TEMP ", in_dist_classes_list_all)

    labels_temp_in_dist = dataset.targets
    # init variables
    list_in_dist_data = []
    list_out_dist_data = []
    # total number of iod observations
    number_iod = int((1 - ood_percentage) * total_unlabeled_obs)
    number_ood = int(ood_percentage * total_unlabeled_obs)
    print("Reading and shuffling data...")
    # list of file names and labels of in distribution data
    #in_dist_classes_list_all = list(np.unique(np.array(labels_temp_in_dist)))
    print("Total number of classes detected: ", len(in_dist_classes_list_all))
    print("List of in distribution classes: ", in_dist_classes_list_all)
    #copy file name and labels to list in data
    for i in range(0, len(list_file_names_in_dist_data)):
        file_name_path = list_file_names_in_dist_data[i][0]
        label_index = labels_temp_in_dist[i]
        #we need to use the actual folder name and not the label index reported by pytorch ImageFolder
        list_in_dist_data += [(file_name_path, in_dist_classes_list_all[label_index])]

    # list of files and labeles out distribution data
    for i in range(0, len(list_file_names_out_dist_data)):
        file_name_path = list_file_names_out_dist_data[i][0]
        list_out_dist_data += [(file_name_path, OOD_LABEL)]
    # shuffle the list and select the percentage of ood and iod data
    random.seed(random_state + batch)
    selected_iod_data = random.sample(list_in_dist_data, number_iod)
    selected_ood_data = random.sample(list_out_dist_data, number_ood)
    print("Number of selected iod observations")
    print(len(selected_iod_data))
    print("Number of selected ood observations")
    print(len(selected_ood_data))
    dest_unlabeled_path_batch = dest_unlabeled_path_base + "/batch_" + str(batch) + "_num_unlabeled_" + str(
        total_unlabeled_obs) + "_ood_perc_" + str(int(100 * ood_percentage)) + "/"
    if (create_dirs):
        # create the directories
        try:
            print("Trying to create directories: ")
            print(dest_unlabeled_path_batch)
            os.makedirs(dest_unlabeled_path_batch)
        except:
            print("Could not create dir, already exists")
    # copy the iid observations
    print("Copying IOD data...")
    # print("The files in the training folder data selected will be deleted...")
    for file_label in selected_iod_data:
        path_src = file_label[0]
        label = random.randint(0, num_classes - 1)
        final_dest = dest_unlabeled_path_batch + "/train/" + str(label) + "/"
        try:
            #print("Trying to create directory: ")
            #print(final_dest)
            os.makedirs(final_dest)
        except:
            a = 0
            #print("Folder already created")
        # print(path_src)
        # print(dest_unlabeled_path_batch)
        copy2(path_src, final_dest)

    # copy the ood observations
    print("Copying OOD data randomly in the training folders...")
    for file_label in selected_ood_data:
        path_src = file_label[0]
        random_label = random.randint(0, num_classes - 1)
        #print("SELECTED LABEL!!! ", random_label)
        #print("FROM ", in_dist_classes_list_all)
        #print("IOD path ", iod_dataset_path)
        #print(random_label)
        file_name = os.path.basename(path_src)
        _, file_extension = os.path.splitext(path_src)
        try:
            #print("Trying to create directory: ")
            #print(final_dest)
            os.makedirs(dest_unlabeled_path_batch + "/train/" + str(random_label))
        except:
            a = 0
        final_dest = dest_unlabeled_path_batch + "/train/" + str(random_label) + "/ood_" + file_name + file_extension

        #print("Folder already created ")
        #print("From: ", path_src)
        #print("To: ", final_dest)
        copy2(path_src, final_dest)
    print("Copying the test folder to the unlabeled data destination... ")
    iod_test_path = (labeled_dataset_path + f"/labeled_batches/batch_{batch}/train").replace("/train","") + "/test/"
    print("From: ", iod_test_path)
    print("To: ", dest_unlabeled_path_batch + "/test/")
    shutil.copytree(iod_test_path, dest_unlabeled_path_batch + "/test/")
    print("A total of ", len(selected_ood_data), " OOD observations were randomly added to the IOD train subfolders!")
    return (number_iod, number_ood)
