import torchvision
import torchvision.transforms as tfms
from PIL import Image as Pili
from statistics import mean
import scipy
from math import log
import torch
from fastai.vision import *
from fastai.callbacks import CSVLogger
from fastai.callbacks.hooks import *
from torchvision import models, transforms
import cv2
from os import listdir
from os.path import isfile, join
import  sklearn.metrics
import pandas as pd
from scipy.stats import entropy

def calculate_mean_std(path_dataset):
    """
    Calculate mean and std of dataset
    :param path_dataset:
    :return:
    """
    dataset = torchvision.datasets.ImageFolder(path_dataset, transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor() ]))
    print(dataset)
    return get_mean_and_std(dataset)

def get_file_names_in_path(path):
    """
    Get files in path
    :param path:
    :return:
    """
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

def pil2fast(img, im_size = 110):
    """
    :param img:
    :param im_size:
    :return:
    """
    data_transform = transforms.Compose(
        [transforms.Resize((im_size, im_size)), transforms.ToTensor()])
    return Image(data_transform(img))

def measure_model_accuracy_test(learner, img_path, class_label, im_size = 110):
    """
    :param learner: fastai model
    :param img_path: image path of test images
    :param class_label: class label of the test images
    :param im_size:
    :return:
    """
    list_images = get_file_names_in_path(img_path)
    print("total of images ", len(list_images))
    num_test = len(list_images)
    correct_preds = 0
    wrong_preds = 0
    for i in range(0, num_test):
        complete_path = img_path + list_images[i]
        image_pil = Pili.open(complete_path).convert('RGB')
        image_fastai = pil2fast(image_pil, im_size=im_size)
        cat_tensor, tensor_class, model_output = learner.predict(image_fastai, with_dropout=False)
        if(tensor_class.item() == class_label):
            correct_preds += 1
        else:
            wrong_preds += 1

    accuracy = correct_preds/num_test
    return accuracy, correct_preds, wrong_preds

def get_precision_recall_f1_score(fastai_model, test_image_path_c0, test_image_path_c1, im_size = 110):
    """
    Calculates precision, recall and f1 score
    :param fastai_model:
    :param test_image_path_c0:
    :param test_image_path_c1:
    :param im_size:
    :return:
    """
    #0 is normal or no pathology
    acc_c0, correct_preds_c0, wrong_preds_c0 = measure_model_accuracy_test(fastai_model, test_image_path_c0,
                                                                          class_label=0,
                                                                          im_size=im_size)
    # 1 is covid-19 positive
    acc_c1, correct_preds_c1, wrong_preds_c1 = measure_model_accuracy_test(fastai_model, test_image_path_c1,
                                                                           class_label=1,
                                                                           im_size=im_size)
    true_positives = correct_preds_c1
    false_positives = wrong_preds_c0
    false_negatives = wrong_preds_c1
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    f1_score = (2 * recall * precision) / (precision + recall)
    return f1_score, recall, precision


def get_metrics_multi_class(learner, list_paths_classes, im_size = 220, name_report_roc = ""):
    """
    Calculates precision, recall and f1 score
    :param fastai_model:
    :param test_image_path_c0:
    :param test_image_path_c1:
    :param im_size:
    :return:
    """
    list_labels = []
    list_predictions = []
    list_scores = []
    number_classes  = len(list_paths_classes)
    calculate_softmax = m = nn.Softmax(dim=0)
    #go through each class to get the array of labels and predictions
    for num_class in range(0, len(list_paths_classes)):
        list_images = get_file_names_in_path(list_paths_classes[num_class])
        #print("total of images ", len(list_images), " for class ", num_class)
        num_test_class = len(list_images)

        #open the images for that class
        for i in range(0, num_test_class):
            complete_path = list_paths_classes[num_class] + list_images[i]
            #print("trying to open ", complete_path)
            image_pil = Pili.open(complete_path).convert('RGB')
            image_fastai = pil2fast(image_pil, im_size=im_size)
            cat_tensor, tensor_class, model_output = learner.predict(image_fastai, with_dropout=False)
            #get score with softmax for ROC curve
            softmaxes = calculate_softmax(model_output)
            selected_softmax = softmaxes[tensor_class.item()].item()
            #store values
            list_predictions += [tensor_class.item()]
            list_scores += [softmaxes[tensor_class.item()].item()]
            list_labels += [num_class]

    #print("len predictions ", len(list_predictions))
    #print("len labels ", len(list_labels))
    list_metrics = []
    list_metrics_names = []

    #calculate metrics using sklearn
    list_metrics += [sklearn.metrics.accuracy_score(list_labels, list_predictions)]
    list_metrics_names += ["accuracy"]
    #precision per class and average
    precisions_classes = sklearn.metrics.precision_score(list_labels, list_predictions, average=None)
    for i in range(0, number_classes):
        list_metrics_names += ["precision class " + str(i)]
        list_metrics += [precisions_classes[i]]
    list_metrics += [mean(precisions_classes)]
    list_metrics_names += ["precision average"]

    #recall per class and average
    recall_classes = sklearn.metrics.recall_score(list_labels, list_predictions, average = None)
    for i in range(0, number_classes):
        list_metrics_names += ["recall class " + str(i)]
        list_metrics += [recall_classes[i]]
    list_metrics += [mean(recall_classes)]
    list_metrics_names += ["recall average"]
    #ROC curve report for binary classification
    if(name_report_roc != "" and number_classes == 2):
        #if its 2 classes only...
        print("processing roc...")
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(list_labels, list_scores)
        print("fpr")
        print(fpr)
        print("tpr")
        print(tpr)
        print("thresholds")
        print(thresholds)
        matrix_roc = np.column_stack((np.array(fpr).flatten(), np.array(tpr).flatten(), np.array(thresholds).flatten()))
        print("saving here: ", name_report_roc)
        np.savetxt(name_report_roc, matrix_roc, delimiter=',')
    #confusion matrix
    confusion_matrix = [sklearn.metrics.confusion_matrix(list_labels, list_predictions)]
    flattened_confusion = sum(confusion_matrix[0].tolist(), [])
    list_metrics_names += ["confusion matrix"]
    list_metrics += flattened_confusion

    return list_metrics, list_metrics_names


def test_output(learn):
    im_size = 110
    img_path_ood = "/media/Data/saul/Datasets/Inbreast_folder_per_class_binary/batch_0/train/0/20586960.bmp"
    image_pil = Pili.open(img_path_ood).convert('RGB')
    image_ood_fastai = pil2fast(image_pil)
    data_transform = transforms.Compose(
        [transforms.Resize((im_size, im_size)), transforms.ToTensor()])
    image_ood_tensor = data_transform(image_pil)
    cat_tensor, tensor_class, model_output = learn.predict(image_ood_fastai, with_dropout=True)
    print("Fastai evaluation output")
    print(model_output)
    # pytorch model
    model = learn.model
    image = torch.zeros(1, 3, im_size, im_size).cuda()
    image[0] = image_ood_tensor
    scores_all_classes = nn.functional.softmax(model(image).data.cpu(), dim=1).squeeze()
    print("Pytorch evaluation output")
    print(scores_all_classes)
    print(model(image).data.cpu())


def get_mean_and_std(dataset):
    """
    Compute the mean and std value of dataset.
    :param dataset:
    :return:
    """
    data_loader = torch.utils.data.DataLoader(dataset,  num_workers= 5, pin_memory=True, batch_size =1)

    #init the mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    k = 1
    for inputs, targets in data_loader:
        #mean and std from the image
        #print("Processing image: ", k)
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
        k += 1

    #normalize
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print("mean: " + str(mean))
    print("std: " + str(std))
    return mean, std