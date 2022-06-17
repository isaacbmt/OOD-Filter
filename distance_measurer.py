import torch
import torchvision.models as models
from fastai.vision import *
# import pingouin as pg
from fastai.callbacks import CSVLogger
from numbers import Integral
import torch
import logging
import sys
from torchvision.utils import save_image
import numpy as np
import pandas as pd
import scipy
from PIL import Image
import torchvision.models.vgg as models2
import torchvision.models as models3


from scipy.stats import entropy
from scipy.spatial import distance
# from utilities.InBreastDataset import InBreastDataset
import matplotlib
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision
from scipy.stats import mannwhitneyu


BATCH_SIZE = 40

def pytorch_feature_extractor():
    input = torch.rand(1, 3, 50, 50)
    vgg16 = models3.resnet152(pretrained=True)
    print(vgg16)
    output = vgg16[:-1](input)
    print(output)


def calculate_Minowski_feature_space_stats(databunch1, databunch2, model, batch_size=BATCH_SIZE, p=2, num_batches=10):
    """
    Calculates the stats for the  Minkowski distance in the feature space, for a given set of batches
    :param databunch1: Dataset 1 to compare
    :param databunch2: Dataset 2 to compare
    :param model: model to extract features
    :param batch_size: batch size
    :param p: 1 or 2, depending upon the usage of Euclidian or Manhattan distances
    :param num_batches: number of batches
    :return:
    """
    run_results = []
    for i in range(0, num_batches):
        dist_i = calculate_Minowski_feature_space(databunch1, databunch2, model, batch_size, p)
        run_results += [dist_i]
    run_results_np = np.array(run_results)
    mean_results = run_results_np.mean()
    std_results = run_results_np.std()
    return (mean_results, std_results)


def calculate_pdf_dist_stats(databunch1, databunch2,  feature_extractor,  batch_size=BATCH_SIZE,
                                        distance_func=distance.jensenshannon, num_batches=10, plot = False, model_name = "densenet"):
    """
    Calculates the stats for the  Probability density functions based  distances in the feature space, for a given set of batches
    :param databunch1: dataset 1
    :param databunch2: dataset 2
    :param feature_extractor:  feature extractor to use
    :param batch_size: batch size to compute
    :param distance_func: pdf distance function to use
    :param num_batches: total number of batches
    :param plot: save plot?
    :param model_name: name or id of the model
    :return:
    """
    run_results = []
    for i in range(0, num_batches):
        dist_i = calculate_pdf_dist(databunch1, databunch2, feature_extractor, batch_size, distance_func, plot, model_name=model_name)
        run_results += [dist_i]
    run_results_np = np.array(run_results)
    mean_results = run_results_np.mean()
    std_results = run_results_np.std()
    return (mean_results, std_results)


def extract_features( feature_extractor, batch_tensors1, model_name = "wideresnet"):
    """
    Extract features from a tensor bunch
    :param feature_extractor: feature extractor to use
    :param batch_tensors1: batch of tensors to project
    :return:
    """

    #print("features bunch 1 dimensions ", features_bunch1.shape)
    if (model_name != "wideresnet"):
        #print("batch tensors shape ", batch_tensors1.shape)
        # the batch might be too large, doing the process in minibatches
        num_observations = batch_tensors1.shape[0]
        mini_batch_size = 20
        num_mini_batches =  int( num_observations / mini_batch_size )
        #print("num mini batches to process ", num_mini_batches)
        for curr_mini_batch in range(1, num_mini_batches + 1):

            batch_tensors = batch_tensors1[(curr_mini_batch - 1) * mini_batch_size : curr_mini_batch * mini_batch_size, :, :]
            #print("batch_tensors minibatch shape ", batch_tensors.shape)
            features_bunch1 = feature_extractor(batch_tensors)
            # pool of non-square window
            #print("features_bunch1 shape ", features_bunch1.shape)
            avg_layer = nn.AvgPool2d((features_bunch1.shape[2], features_bunch1.shape[3]), stride=(1, 1))
            #averaging the features to lower the dimensionality in case is not wide resnet
            #print("features_bunch1  MAKE SMALLER! ", features_bunch1.shape)
            features_bunch1 = avg_layer(features_bunch1)
            features_bunch1 = features_bunch1.view(-1, features_bunch1.shape[1] * features_bunch1.shape[2] *
                                                   features_bunch1.shape[3])
            #print("features_bunch1 dimensions")
            #print(features_bunch1.shape)
            #concatenate features
            if(curr_mini_batch == 1):
                features_bunch = features_bunch1
                #print("features_bunch first ", features_bunch.shape)
            else:
                features_bunch = torch.cat((features_bunch, features_bunch1), 0)
    else:
        features_bunch = feature_extractor(batch_tensors1)

    #print("features_bunch ", features_bunch.shape)
    return features_bunch

def calculate_pdf_dist(databunch1, databunch2, feature_extractor, batch_size=BATCH_SIZE, distance_func=distance.jensenshannon, plot = False, model_name = "alexnet" ):
    """
    Calculates the probability density function distance in the feature space
    :param databunch1: dataset 1
    :param databunch2: dataset 2
    :param feature_extractor:  feature extractor to use
    :param batch_size: batch size to compute
    :param distance_func: distance function to use
    :param plot: plot the feature densities?
    :param model_name: model id or name?
    :return:
    """
    # just get the number of dimensions

    tensorbunch1 = databunch_to_tensor(databunch1)
    tensorbunch2 = databunch_to_tensor(databunch2)

    batch_tensors1 = tensorbunch1[0:batch_size, :, :, :]
    #print("batch_tensors1 shape before ", batch_tensors1.shape)
    # get number of features
    features_bunch1 = extract_features(feature_extractor, batch_tensors1, model_name = model_name)
    num_features = features_bunch1.shape[1]
    #print("Calculating pdf distance for for feature space of dimensions: ", num_features)
    js_dist_dims = []
    # calculate distance of histograms for given
    print("Calculating the histogram for all the  ", num_features, " features ")
    for i in range(0, num_features):
        js_dist_dims += [
            calculate_distance_hists(tensorbunch1, tensorbunch2, feature_extractor, dimension=i, batch_size=batch_size,
                                     distance_func=distance_func, model_name = model_name, plot= plot)]
    js_dist_sum = sum(js_dist_dims)
    return js_dist_sum


def calculate_distance_hists(tensorbunch1, tensorbunch2, feature_extractor, dimension, batch_size=BATCH_SIZE,
                             distance_func=distance.jensenshannon, plot = False, model_name = "wideresenet"):
    """
    Calculates the distance between two histograms of features
    :param tensorbunch1: Databunch 1 to compare
    :param tensorbunch2: Databunch 2 to compare
    :param feature_extractor: Feature extractor from the model to use to compare
    :param dimension: The given dimension where to calculate the histogram
    :param batch_size: Batch size of the random set of observations where to calculate the histogram
    :param distance_func: the given distance to use
    :param plot: Plot the histogram?
    :param model_name: Model name
    :return:
    """
    # random pick of batch observations
    total_number_obs_1 = tensorbunch1.shape[0]
    total_number_obs_2 = tensorbunch2.shape[0]
    batch_indices_1 = generate_rand_bin_array(batch_size, total_number_obs_1)
    batch_indices_2 = generate_rand_bin_array(batch_size, total_number_obs_2)
    # create the batch of tensors to get its features
    batch_tensors1 = tensorbunch1[batch_indices_1, :, :, :]
    batch_tensors2 = tensorbunch2[batch_indices_2, :, :, :]
    
    # get the  features from the selected batch
    features_bunch1 = extract_features(feature_extractor, batch_tensors1, model_name= model_name)
    values_dimension_bunch1 = features_bunch1[:, dimension].cpu().detach().numpy()
    del features_bunch1


    features_bunch2 = extract_features(feature_extractor, batch_tensors2, model_name= model_name)
    
    # get the values of a specific dimension
    values_dimension_bunch2 = features_bunch2[:, dimension].cpu().detach().numpy()
    # calculate the histograms
    (hist1, bucks1) = np.histogram(values_dimension_bunch1, bins=15, range=None, normed=None, weights=None,
                                   density=None)
    #ensure that the histograms have the same meaning, by using the same buckets
    (hist2, bucks2) = np.histogram(values_dimension_bunch2, bins=bucks1, range=None, normed=None, weights=None,
                                   density=None)

    # normalize the histograms
    hist1 = np.array(hist1) / sum(hist1)
    if sum(hist2) != 0:
        hist2 = np.array(hist2) / sum(hist2)
    else:
        hist2 = np.array(len(hist2.tolist()) * [0.0000001])

    # ent = entropy(hist1.tolist(), qk = hist2.tolist())
    # print(hist1.tolist())
    # print(hist2.tolist())
    js_dist = distance_func(hist1.tolist(), hist2.tolist())
    # js_dist = distance_func(hist1.tolist(), hist2.tolist(),2.0)
    # jensen shannon distance of the histograms from the same dimension

    # print(js_dist)
    integer_part = int(js_dist)
    float_part = int((js_dist - float(integer_part))*10000)
    #plot_name = "plots/" + str(integer_part) + "_" + str(float_part) + "_js_dist_" +  str(dimension) + "_feature_num.pdf"
    #plot_histogram(bucks1, hist1, bucks2, hist2, plot_name)
    return js_dist




def plot_histogram(bins1, y1, bins2, y2,  plot_name = "histogram.pgf"):
    """
    Histogram plotter in latex, saves the plot to latex
    :param bins1: bins of the first histogram
    :param y1: occurrences for the first plot
    :param bins2:bins of the second histogram
    :param y2: occurrences for the second plot
    :param plot_name: plot name
    :param title_plot: plot's title
    :return:
    """
    #matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'font.size': 20
    })
    print("hist1 ")
    print(y1.shape)
    print("buckets ")
    print(bins1[0:-1].shape)
    fig, ax = plt.subplots()
    ax.plot(bins1[0:-1], y1/y1.sum(), '--')
    ax.plot(bins2[0:-1], y2 / y2.sum(), '-x')
    ax.set_xlabel('Feature values')
    ax.set_ylabel('Probability density')
    #ax.set_title(title_plot)

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    print("Saving fig with name ")
    print(plot_name)
    plt.savefig(plot_name, dpi=400)

# databunch1 is the smallest
def calculate_Minowski_feature_space(databunch1, databunch2, model, batch_size = BATCH_SIZE, p = 2):
    """
    Calculates the Minowski distance in the feature space, for two databunches
    :param databunch1: dataset 1
    :param databunch2: dataset 2
    :param model: model to do the projection
    :param batch_size: batch size
    :param p: 1 or 2, if Manhattan or Euclidian distance is used
    :return:
    """
    print("Calculating Minowski distance of two samples of two datasets, p: ", p)
    feature_extractor = get_feature_extractor(model)
    # print("Observation 1 shape")
    # print(len(databunch1.train_ds))
    tensorbunch1 = databunch_to_tensor(databunch1)
    tensorbunch2 = databunch_to_tensor(databunch2)
    # get the randomized batch indices
    total_number_obs_1 = tensorbunch1.shape[0]
    total_number_obs_2 = tensorbunch2.shape[0]
    batch_indices_1 = generate_rand_bin_array(batch_size, total_number_obs_1)
    batch_indices_2 = generate_rand_bin_array(batch_size, total_number_obs_2)
    # print("batch indices")
    # print(batch_indices_1.shape)
    # print(batch_indices_1.sum())

    # total number of observations of the smallest databunch
    total_observations_bunch1 = tensorbunch1.shape[0]
    # print("Tensor bunch with dimensions ")
    # print(tensorbunch1.shape)
    # pick random observations for the batch
    batch_tensors1 = tensorbunch1[batch_indices_1, :, :, :].to(device="cuda:0")
    batch_tensors2 = tensorbunch2[batch_indices_2, :, :, :].to(device="cuda:0")
    # print("batch tensors dims")
    # print(batch_tensors1.shape)
    # extract its features
    features_bunch1 = feature_extractor(batch_tensors1)
    features_bunch2 = feature_extractor(batch_tensors2)

    sum_mses = []
    # one to all distance accumulation
    for i in range(0, batch_size):
        mse_i = calculate_Minowski_observation_min(features_bunch1[i], features_bunch2, p)
        sum_mses += [mse_i.item()]
    sum_mses_np = np.array(sum_mses)
    # delete features to prevent gpu memory overflow
    del features_bunch1
    del features_bunch2
    torch.cuda.empty_cache()
    mse_mean_all_batch = sum_mses_np.mean()
    # take one batch
    return mse_mean_all_batch


def calculate_Minowski_observation(observation, tensorbunch, p=2):
    """
    Calculates the Minowski distance for a given observation, to a given dataset bunch, using the average as final measure
    :param observation: tensor with individual observation
    :param tensorbunch: set of observations to compare with
    :param p: 1 or 2, Manhattan and Euclidian distances
    :return:
    """
    # vectorize all the images in tensorbunch
    # if it receives an image
    # tensorbunch_vec = img[:].view(-1, tensorbunch.shape[1]*tensorbunch.shape[2]*tensorbunch.shape[3])
    observation_vec = observation.view(-1)
    # difference between bunch of tensors and the current observation to analyze
    difference_bunch = tensorbunch - observation_vec
    # for all observations in the bunch, calculate its euclidian proximity
    # L2 Norm over columns
    minowski_distances = torch.norm(difference_bunch, p, 1)
    # choose mse or min?
    minowski_distance = minowski_distances.sum() / len(minowski_distances)
    return minowski_distance


def calculate_Minowski_observation_min(observation, tensorbunch, p = 2):
    """
    Calculates the Minowski distance for a given observation, to a given dataset bunch, using the minimum distance  as final measure
    :param observation: tensor with individual observation
    :param tensorbunch: set of observations to compare with
    :param p: 1 or 2, Manhattan and Euclidian distances
    :return:
    """
    # vectorize all the images in tensorbunch
    # if it receives an image
    # tensorbunch_vec = img[:].view(-1, tensorbunch.shape[1]*tensorbunch.shape[2]*tensorbunch.shape[3])
    observation_vec = observation.view(-1)
    # difference between bunch of tensors and the current observation to analyze
    difference_bunch = tensorbunch - observation_vec
    # for all observations in the bunch, calculate its euclidian proximity
    # L2 Norm over columns
    minowski_distances = torch.norm(difference_bunch, p, 1)
    # choose mse or min?
    min_dist = minowski_distances.min()
    return min_dist


def databunch_to_tensor(databunch1):
    """
    Translates a databunch (fastai) to tensor
    :param databunch1:
    :return:
    """
    # tensor of tensor
    tensor_bunch = torch.zeros(len(databunch1.train_ds), databunch1.train_ds[0][0].shape[0],
                               databunch1.train_ds[0][0].shape[1], databunch1.train_ds[0][0].shape[2], device="cuda:0")
    for i in range(0, len(databunch1.train_ds)):
        # print(databunch1.train_ds[i][0].data.shape)
        tensor_bunch[i, :, :, :] = databunch1.train_ds[i][0].data.to(device="cuda:0")

    return tensor_bunch


def get_feature_extractor(model, model_name = "alexnet"):
    """
    Gets the feature extractor from a given model
    :param model: model to use as feature extractor
    :return: returns the feature extractor which can be used later
    """
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path)
    # save learner to reload it as a pytorch model
    learner = Learner(data, model, metrics=[accuracy])
    # CHANGE!
    path_temp = "/content/utilities/model/final_model_.pk"
    learner.export(path_temp)
    torch_dict = torch.load(path_temp)
    # get the model
    model_loaded = torch_dict["model"]
    # put it on gpu!
    model_loaded = model_loaded.to(device="cuda:0")
    model_loaded.eval()
    # usually the last set of layers act as classifier, therefore we discard it
    if (model_name == "wideresnet"):
        feature_extractor = model_loaded.features[:-1]
        print("Using wideresenet")

    if (model_name == "alexnet"):
        feature_extractor = model_loaded.features[:-1]
        print("Using alexnet")

    if (model_name == "densenet"):
        # print(model_loaded.features)
        feature_extractor = model_loaded.features[:-2]
        print("Using densenet")
    return feature_extractor



def dataset_distance_tester_pdf(path_bunch1 = "/media/Data/saul/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_", 
                                path_bunch2 = "/media/Data/saul/Datasets/MNIST_medium_complete/batches_unlabeled/batch_",
                                ood_perc = 100, num_unlabeled = 3000, name_ood_dataset = "in_dist",
                                num_batches=1, size_image = 220, distance_func = distance.jensenshannon, 
                                plot = False, batch_size_p = 40, model_name="alexnet", filtered=False, reports_path=""):
    """
    Dataset distance tester for pdf distances
    :param path_bunch1:
    :param path_bunch2:
    :param ood_perc:
    :param num_unlabeled:
    :param name_ood_dataset:
    :param num_batches:
    :param size_image:
    :param distance_func:
    :param plot:
    :param batch_size_p:
    :param model_name:
    :return:
    """

    global key
    key = "pdf"
    print("Computing distance for dataset: ", name_ood_dataset)
    if(model_name == "wideresnet"):
        model = models.WideResNet(num_groups=3, N=4, num_classes=10, k=2, start_nf=64)
        print("Using wideresenet")
    elif(model_name == "alexnet"):
#         model = models.densenet121(pretrained=True)
        model = models.alexnet(pretrained=True)
        print("using alexnet")
    elif (model_name == "densenet"):
        model = models.densenet121(num_classes=10)
        print("using densenet121")
    dists_reference = []
    dists_bunch1_bunch2 = []
    dists_substracted = []
    feature_extractor = get_feature_extractor(model, model_name=model_name)
    for i in range(0, num_batches):
        #Be sure to take only train data!
        path_mnist_half_in_dist = path_bunch1 + "/batch_" + str(i) + "/train"
        if filtered:
            path_mnist_half_out_dist = path_bunch2 + "/batch_" + str(i) + "/batch_" + str(i) + "_num_unlabeled_" + str(num_unlabeled) + "_ood_perc_" + str(ood_perc) + "/train/"
        else:
            path_mnist_half_out_dist = path_bunch2 + "/batch_" + str(i) + "_num_unlabeled_" + str(num_unlabeled) + "_ood_perc_" + str(ood_perc) + "/train/"
        print("LOADING IN DIST PATH ", path_mnist_half_in_dist)

        databunch1 = (ImageList.from_folder(path_mnist_half_in_dist)
                    .split_none()
                    .label_from_folder()
                    .transform(size=size_image)
                    .databunch())
        print("LOADING OUT DIST PATH ", path_mnist_half_out_dist)
        databunch2 = (ImageList.from_folder(path_mnist_half_out_dist)
                    .split_none()
                    .label_from_folder()
                    .transform(size=size_image)
                    .databunch())


        (dist_ref_i, std_ref) = calculate_pdf_dist_stats(databunch1, databunch1,  feature_extractor= feature_extractor, batch_size=batch_size_p,
                                        distance_func=distance_func, num_batches=3, model_name=model_name)
        dists_reference += [dist_ref_i]
        print("Distance to itself    (reference): ", dist_ref_i, " for batch: ", i)
        (dist_between_bunches_i, dist_between_bunches_std) = calculate_pdf_dist_stats(databunch1, databunch2, feature_extractor= feature_extractor, batch_size=batch_size_p, distance_func=distance_func, num_batches=3, plot = plot, model_name=model_name)
        dists_bunch1_bunch2 += [dist_between_bunches_i]
        print("Distance between bunches:  ", dist_between_bunches_i, " for batch:", i)
        dists_substracted += [abs(dist_between_bunches_i - dist_ref_i)]

    dist_between_bunches = np.mean(dists_substracted)
    print("Distance  between bunches: ", dist_between_bunches)
    stat, p_value = scipy.stats.wilcoxon(dists_reference, dists_bunch1_bunch2, correction = True)
    #means
    dists_reference += [np.array(dists_reference).mean()]
    dists_bunch1_bunch2 += [np.array(dists_bunch1_bunch2).mean()]
    dists_substracted += [np.array(dists_substracted).mean()]
    # stds are the last row
    dists_reference += [np.array(dists_reference).std()]
    dists_bunch1_bunch2 += [np.array(dists_bunch1_bunch2).std()]
    dists_substracted += [np.array(dists_substracted).std()]

    header3 = 'Distance_substracted with p ' + str(p_value)

    dict_csv = {'Reference': dists_reference,
              'Distance': dists_bunch1_bunch2,
              header3: dists_substracted

              }
    dataframe = pd.DataFrame(dict_csv, columns=['Reference', 'Distance', header3])
    dataframe.to_csv(fr'{reports_path}/' + name_ood_dataset + "ood_perc_" + str(ood_perc) + '.csv', index=False, header=True)

    return dist_between_bunches

    #calculate distance

    dist2 = calculate_Minowski_feature_space_stats(databunch1, databunch2, model, batch_size=BATCH_SIZE, p=2, num_batches=3)
    print("Distance MNIST in dist to MNIST out dist : ", dist2)
    reference2 = calculate_Minowski_feature_space_stats(databunch1, databunch1, model, batch_size=BATCH_SIZE, p=2, num_batches=3)
    print("Distance MNIST in dist to MNIST out dist (second): ", reference2)



def dataset_distance_tester(path_bunch1 = "/media/Data/saul/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_", path_bunch2 = "/media/Data/saul/Datasets/MNIST_medium_complete/batches_unlabeled/batch_",ood_perc = 100, num_unlabeled = 3000, name_ood_dataset = "in_dist", num_batches=10, size_image = 28, p = 2):
    """
    Dataset distance calculator
    :param path_bunch1:
    :param path_bunch2:
    :param ood_perc:
    :param num_unlabeled:
    :param name_ood_dataset:
    :param num_batches:
    :param size_image:
    :param p:
    :return:
    """
    global key
    key = "minkowski"


    print("Computing distance for dataset: ", name_ood_dataset, " p: ", p, " ood: ", ood_perc)
    model = models.WideResNet(num_groups=3, N=4, num_classes=10, k=2, start_nf=64)
    dists_reference = []
    dists_bunch1_bunch2 = []
    dists_substracted = []
    for i in range(1, num_batches):
        path_mnist_half_in_dist = path_bunch1 + str(i)
        path_mnist_half_out_dist = path_bunch2 + str(i) + "/batch_" + str(i) + "_num_unlabeled_" + str(num_unlabeled) + "_ood_perc_" + str(ood_perc)

        databunch1 = (ImageList.from_folder(path_mnist_half_in_dist)
                    .split_none()
                    .label_from_folder()
                    .transform(size=size_image)
                    .databunch())
        databunch2 = (ImageList.from_folder(path_mnist_half_out_dist)
                    .split_none()

                    .label_from_folder()
                    .transform(size=size_image)
                    .databunch())

        #databunch1 = ImageDataBunch.from_folder(path_mnist_half_in_dist, ignore_empty=True)
        #databunch2 = ImageDataBunch.from_folder(path_mnist_half_out_dist, ignore_empty=True)
        (dist_ref_i, std_ref) = calculate_Minowski_feature_space_stats(databunch1, databunch1, model, batch_size=BATCH_SIZE, p=p, num_batches=3)
        dists_reference += [dist_ref_i]
        print("Distance to itself    (reference): ", dist_ref_i, " for batch: ", i)
        (dist_between_bunches_i, dist_between_bunches_std) = calculate_Minowski_feature_space_stats(databunch1, databunch2, model, batch_size=BATCH_SIZE, p=p, num_batches=3)
        dists_bunch1_bunch2 += [dist_between_bunches_i]
        print("Distance between bunches:  ", dist_between_bunches_i, " for batch:", i)
        dists_substracted += [abs(dist_between_bunches_i - dist_ref_i)]




    dist_between_bunches = np.mean(dists_substracted)
    print("Distance  between bunches: ", dist_between_bunches)
    stat, p_value = scipy.stats.wilcoxon(dists_reference, dists_bunch1_bunch2, correction = True)
    #means
    dists_reference += [np.array(dists_reference).mean()]
    dists_bunch1_bunch2 += [np.array(dists_bunch1_bunch2).mean()]
    dists_substracted += [np.array(dists_substracted).mean()]
    # stds are the last row
    dists_reference += [np.array(dists_reference).std()]
    dists_bunch1_bunch2 += [np.array(dists_bunch1_bunch2).std()]
    dists_substracted += [np.array(dists_substracted).std()]

    header3 = 'Distance_substracted with p ' + str(p_value)

    dict_csv = {'Reference': dists_reference,
              'Distance': dists_bunch1_bunch2,
              header3: dists_substracted

              }
    dataframe = pd.DataFrame(dict_csv, columns=['Reference', 'Distance', header3])
    dataframe.to_csv(r'C:\Users\XT\Documents\OOD4SSDL_clean\Distance_reports/' + name_ood_dataset + "_p_" + str(p)+ "_OOD_perc_" + str(ood_perc) +'.csv', index=False, header=True)

    return dist_between_bunches



    #calculate distance

    dist2 = calculate_Minowski_feature_space_stats(databunch1, databunch2, model, batch_size=BATCH_SIZE, p=2, num_batches=3)
    print("Distance MNIST in dist to MNIST out dist : ", dist2)
    reference2 = calculate_Minowski_feature_space_stats(databunch1, databunch1, model, batch_size=BATCH_SIZE, p=2, num_batches=3)
    print("Distance MNIST in dist to MNIST out dist (second): ", reference2)


def generate_rand_bin_array(num_ones, length_array):
    arr = np.zeros(length_array)
    arr[:num_ones] = 1
    np.random.shuffle(arr)
    bool_array = torch.tensor(array(arr.tolist(), dtype=bool))
    return bool_array
