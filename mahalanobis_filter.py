import os
import torch
from fastai.vision import *
import torchvision.models as models
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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

from scipy.stats import entropy
from scipy.spatial import distance
# from utilities.InBreastDataset import InBreastDataset
import matplotlib
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
import torchvision
from scipy.stats import mannwhitneyu
#from shutil import copyfile
import shutil
torch.set_printoptions(threshold=10_000)

BATCH_SIZE = 4


class OOD_filter_Mahalanobis:
    def __init__(self, model_name = "wideresnet", num_classes = 2, thresholding_method="real"):
        """
        OOD filter constructor
        :param model_name: name of the model to get the feature space from, pretrained with imagenet, wideresnet and densenet have been tested so far
        """
        self.model_name = model_name
        self.thresholding_method = thresholding_method
        #list of scores, filepaths and labels of unlabeled data processed
        self.scores = []
        self.file_paths = []
        self.labels = []
        self.file_names = []
        #was copied, was test?
        self.info = []
        self.num_classes = num_classes
        self.est_clusters_number = []
        self.real_cluster_number = []

    def get_Mahalanobis_distance(self, Covariance_mat_pseudoinverse, features_obs_batch, mean_features_all_observations):
        """
        Calculate the Mahalanbis distance for a features obs batch of 1
        :param Covariance_mat:
        :param features_obs_batch: Only 1 observation is supported
        :param mean_features_all_observations:
        :return:
        """
        # substract the mean to the current observation
        fac_quad_form = features_obs_batch - mean_features_all_observations
        fac_quad_form_t = fac_quad_form.transpose(0, 1)
        # evaluate likelihood for all observations
        likelihood_batch = fac_quad_form.mm(Covariance_mat_pseudoinverse).mm(fac_quad_form_t).item()
        return likelihood_batch

    def extract_features(self, feature_extractor, batch_tensors1):
        """
        Extract features from a tensor bunch
        :param feature_extractor:
        :param batch_tensors1:
        :return:
        """
        features_bunch1 = feature_extractor(batch_tensors1)
        if (self.model_name != "wideresnet"):

            # pool of non-square window
            #print("features_bunch1 shape ", features_bunch1.shape)
            avg_layer = nn.AvgPool2d((features_bunch1.shape[2], features_bunch1.shape[3]), stride=(1, 1))
            #averaging the features to lower the dimensionality in case is not wide resnet
            features_bunch1 = avg_layer(features_bunch1)
            features_bunch1 = features_bunch1.view(-1, features_bunch1.shape[1] * features_bunch1.shape[2] *
                                                   features_bunch1.shape[3])
            print("features_bunch1 dimensions")
            print(features_bunch1.shape)

        return features_bunch1

    def calculate_covariance_matrix_pseudoinverse(self, tensorbunch1, feature_extractor, dimensions, batch_size=5, num_bins=15, plot=False):
        """
        Returns the pseudo inverse of the cov matrix
        :param tensorbunch1:
        :param feature_extractor:
        :param dimensions:
        :param batch_size:
        :param num_bins:
        :param plot:
        :return:
        """
        print("Number of observations")
        total_number_obs_1 = tensorbunch1.shape[0]
        print("Number of dimensions ", dimensions)
        features_all_observations = torch.zeros((total_number_obs_1, dimensions), device="cuda:0")

        # print("total number of obs ", total_number_obs_1)
        number_batches = total_number_obs_1 // batch_size
        batch_tensors1 = tensorbunch1[0: batch_size, :, :, :]
        # get the  features from the selected batch
        features_bunch1 = self.extract_features(feature_extractor, batch_tensors1)
        # for each dimension, calculate its histogram
        # get the values of a specific dimension
        features_all_observations = features_bunch1[:, :].cpu().detach().numpy()
        # Go through each batch...
        for current_batch_num in range(1, number_batches):
            # create the batch of tensors to get its features
            batch_tensors1 = tensorbunch1[(current_batch_num) * batch_size: (current_batch_num + 1) * batch_size, :, :,:]
            # get the  features from the selected batch
            features_bunch1 = self.extract_features(feature_extractor, batch_tensors1)
            # get the values of a specific dimension
            values_dimension_bunch1 = features_bunch1[:, :].cpu().detach().numpy()
            features_all_observations = np.concatenate((features_all_observations, values_dimension_bunch1), 0)
        features_all_observations = features_all_observations.transpose()
        mean_features_all_observations = torch.mean(torch.tensor(features_all_observations, device="cuda:0", dtype = torch.float), 1)
        #calculate the covariance matrix
        print(features_all_observations)
        Covariance_mat = np.cov(features_all_observations)
        print(Covariance_mat)
        # Covariance_matrix_pinv = torch.tensor(Covariance_mat, dtype=torch.float, device="cuda:0")
        Covariance_matrix_pinv = torch.tensor(np.linalg.pinv(Covariance_mat), dtype=torch.float, device="cuda:0")
        print(Covariance_matrix_pinv)
        #return the cov mat as a tensor
        Covariance_matrix_pinv_torch = torch.tensor(Covariance_matrix_pinv, device="cuda:0", dtype = torch.float)


        return Covariance_matrix_pinv_torch, mean_features_all_observations

    def plot_histogram(self, bins1, y1, plot_name="histogram.pgf"):
        """
        Histogram plotter in latex, saves the plot to latex
        :param bins1:
        :param y1:
        :param bins2:
        :param y2:
        :param plot_name:
        :param title_plot:
        :return:
        """

        # matplotlib.use("pgf")
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
        ax.plot(bins1[0:-1], y1 / y1.sum(), '--')

        ax.set_xlabel('Feature values')
        ax.set_ylabel('Probability density')
        # ax.set_title(title_plot)

        # Tweak spacing to prevent clipping of ylabel

        fig.tight_layout()
        print("Saving fig with name ")
        print(plot_name)
        plt.savefig(plot_name, dpi=400)

    def databunch_to_tensor(self, databunch1):
        """
        Convert the databunch to tensor set
        :param databunch1: databunch to convert
        :return: converted tensor
        """
        # tensor of tensor
        print("pegado")
        tensor_bunch = torch.zeros(len(databunch1.train_ds), databunch1.train_ds[0][0].shape[0],
                                   databunch1.train_ds[0][0].shape[1], databunch1.train_ds[0][0].shape[2],
                                   device="cuda:0")
        print("hola")
        for i in range(0, len(databunch1.train_ds)):
            # print(databunch1.train_ds[i][0].data.shape)
            tensor_bunch[i, :, :, :] = databunch1.train_ds[i][0].data.to(device="cuda:0")

        print("Finish databunch to tensor")
        return tensor_bunch

    def get_feature_extractor(self, model):
        """
        Gets the feature extractor from a given model
        :param model: model to use as feature extractor
        :return: returns the feature extractor which can be used later
        """
        path = untar_data(URLs.MNIST_TINY)
        data = ImageDataBunch.from_folder(path)
        # save learner to reload it as a pytorch model
        print("Learner...")
        learner = Learner(data, model, metrics=[accuracy])
        #CHANGE!
        path_temp="utilities/model/final_model_.pk"
        learner.export(path_temp)
        torch_dict = torch.load(path_temp)
        # get the model
        model_loaded = torch_dict["model"]
        # put it on gpu!
        model_loaded = model_loaded.to(device="cuda:0")
        model_loaded.eval()
        # usually the last set of layers act as classifier, therefore we discard it
        if(self.model_name == "wideresnet"):
            print(model_loaded)
            feature_extractor = model_loaded.features[:-1]
            print("Using wideresenet")

        if (self.model_name == "alexnet"):
            feature_extractor = model_loaded.features[:-1]
            print("Using alexnet")

        if(self.model_name == "densenet"):
            #print(model_loaded.features)
            feature_extractor = model_loaded.features[:-2]
            print("Using densenet")
        return feature_extractor

    def run_filter(self, path_bunch1, path_bunch2, ood_perc=100, num_unlabeled=3000,
                   num_batches=10, size_image=120, batch_size_p=BATCH_SIZE,
                   dir_filtered_root="/media/Data/saul/Datasets/Covid19/Dataset/OOD_COVID19/OOD_FILTERED/batch_",
                   ood_thresh=0.8, path_reports_ood="/reports_ood/"):
        """
        :param path_bunch1: path for the first data bunch, labeled data
        :param path_bunch2: unlabeled data
        :param ood_perc: percentage of data ood in the unlabeled dataset
        :param num_unlabeled: number of unlabeled observations in the unlabeled dataset
        :param name_ood_dataset: name of the unlabeled dataset
        :param num_batches: Number of batches of the unlabeled dataset to filter
        :param size_image: input image dimensions for the feature extractor
        :param batch_size_p: batch size
        :param dir_filtered_root: path for the filtered data to be stored
        :param ood_thresh: ood threshold to apply
        :param path_reports_ood: path for the ood filtering Reports
        :return:
        """
        #Should be one for the Mahalanobis distance to work
        if ood_thresh == 0 or ood_thresh == 1:
            self.real_cluster_number = [1] * num_batches
        else:
            self.real_cluster_number = [2] * num_batches

        batch_size_unlabeled = 1
        batch_size_labeled = 10
        global key
        key = "pdf"
        print("Filtering OOD data for dataset at: ", path_bunch2)
        print("OOD threshold ", ood_thresh)
        for num_batch_data in range(0, num_batches):
            # load pre-trained model, CORRECTION
            # model = models.alexnet(pretrained=True)
            if(self.model_name == "wideresnet"):
                print("Using wideresnet")
                model = models.wide_resnet50_2(pretrained=True)
#                 model = models.WideResNet(num_groups=3, N=4, num_classes=10, k=2, start_nf=64)
            if (self.model_name == "densenet"):
                print("Using densenet")
                model = models.densenet121(pretrained=True) #(num_classes=10)
            if (self.model_name == "alexnet"):
                print("alexnet")
                model = models.alexnet(pretrained=True)
#                 model = models.alexnet(num_classes=10)
            # number of histogram bins
            num_bins = 15
            print("Processing batch of labeled and unlabeled data: ", num_batch_data)
            # paths of data for all batches
            #DEBUG INCLUDE TRAIN
            path_labeled = path_bunch1 + "/batch_" + str(num_batch_data) + "/train/"
            path_unlabeled = path_bunch2 + "/batch_" + str(num_batch_data) + "_num_unlabeled_" + str(
                num_unlabeled) + "_ood_perc_" + str(ood_perc)
            print("path labeled ", path_labeled)
            print("path unlabeled ", path_unlabeled)
            # get the dataset readers
            #  S_l
            databunch_labeled = (ImageList.from_folder(path_labeled)
                                 .split_none()
                                 .label_from_folder()
                                 .transform(size=size_image)
                                 .databunch())
            # S_u
            databunch_unlabeled = (ImageList.from_folder(path_unlabeled)
                                   .split_none()
                                   .label_from_folder()
                                   .transform(size=size_image)
                                   .databunch())
            # path_unlabeled = path_bunch2 + "/batch_" + str(num_batch_data) + "_num_unlabeled_" + str(
            #     num_unlabeled) + "_ood_perc_" + str(ood_perc) + "/train/"
            # databunch_unlabeled_no_test = (ImageList.from_folder(path_unlabeled)
            #                        .split_none()
            #                        .label_from_folder()
            #                        .transform(size=size_image)
            #                        .databunch())
            # print(databunch_unlabeled_no_test)
            # get tensor bunches
            tensorbunch_labeled = self.databunch_to_tensor(databunch_labeled)
            tensorbunch_unlabeled = self.databunch_to_tensor(databunch_unlabeled)
            # shuffle batch
            
            num_obs_unlabeled = tensorbunch_unlabeled.shape[0]
            num_obs_labeled = tensorbunch_labeled.shape[0]
            print("Number of unlabeled observations in batch: ", num_obs_unlabeled)
            print("Number of  labeled observations in batch: ", num_obs_labeled)
            # calculate the number of batches
            num_batches_unlabeled = num_obs_unlabeled // batch_size_unlabeled
            print("Number of unlabeled data batches to process: ", num_batches_unlabeled)
            # DO THIS FOR ALL THE BATCHES
            # get number of features
            batch_tensors1 = tensorbunch_labeled[0:batch_size_p, :, :, :]
            
            feature_extractor = self.get_feature_extractor(model)
            # print(f"batch_tensors1 device: {batch_tensors1.get_device()}")
            # print(f"feature extractor device: {next(feature_extractor).parameters}")
            features_bunch1 = self.extract_features(feature_extractor, batch_tensors1)
            num_features = features_bunch1.shape[1]
            print("Number of features: ", num_features)
            # go through each batch unlabeled
            gauss_likelihoods_final_all_obs = []
            print("Calculating the covariance matrix from the labeled data...")
            Covariance_mat_pseudoinverse, mean_features_all_observations = self.calculate_covariance_matrix_pseudoinverse(tensorbunch_labeled, feature_extractor, num_features, batch_size=batch_size_labeled, num_bins=num_bins, plot=False)
            print("Cov shape: ", Covariance_mat_pseudoinverse.shape)
            print("Evaluating Mahalanobis distance for unlabeled data...")
            for current_batch_num_unlabeled in range(0, num_batches_unlabeled):
                # print("Calculating pdf distance for for feature space of dimensions: ", num_features)
                # get the features for the current unlabeled batch, CORRECTION Batch_number=current_batch_num_unlabeled
                values_features_bunch_unlabeled, batch_indices_unlabeled = self.get_batch_features(tensorbunch_unlabeled,
                                                                                              batch_size_unlabeled=batch_size_unlabeled,
                                                                                              batch_number=current_batch_num_unlabeled,
                                                                                              feature_extractor=feature_extractor)
                num_obs_unlabeled_batch = values_features_bunch_unlabeled.shape[0]
                # init buffer with dims
                likelihoods_all_obs_all_dims = torch.zeros(num_obs_unlabeled_batch, num_features)
                # go  through each dimension, and calculate the likelihood for the whole unlabeled dataset
                likelihoods_gauss_batch = self.get_Mahalanobis_distance(Covariance_mat_pseudoinverse, values_features_bunch_unlabeled, mean_features_all_observations)

                # calculate the log of the sum of the likelihoods for all the dimensions, obtaining a score per observation
                #THE LOWER THE BETTER
                # store the likelihood for all the observations
                gauss_likelihoods_final_all_obs += [likelihoods_gauss_batch]
            self.create_report_and_filtered_folder(gauss_likelihoods_final_all_obs, databunch_unlabeled,
                                              dir_filtered_root, num_batch_data, num_unlabeled, ood_perc,
                                              path_reports_ood, ood_thresh)
        if self.thresholding_method != "real" and self.thresholding_method != "dbscan":
            print("r: ", self.real_cluster_number)
            print("est: ", self.est_clusters_number)
            dict_csv = {"expected_clusters": self.real_cluster_number,
                        "estimated_clusters": self.est_clusters_number}
            df = pd.DataFrame(dict_csv, columns=["expected_clusters", "estimated_clusters"])
            df.to_csv(path_reports_ood + "estimated_clusters.csv", index=False, header=True)



    def create_report_and_filtered_folder(self, gauss_likelihoods_final_all_obs, databunch_unlabeled, dir_filtered_root, num_batch_data, num_unlabeled, ood_perc, path_reports_ood, ood_thresh):
        """
        Create the report of OOD data and store the filtered data
        :param gauss_likelihoods_final_all_obs:
        :param databunch_unlabeled:
        :param dir_filtered_root:
        :param num_batch_data:
        :param num_unlabeled:
        :param ood_perc:
        :param path_reports_ood:
        :return:
        """
        # store per file scores
        file_names = []
        file_paths = []
        scores = []
        labels = []
        # create final summary
        for j in range(0, len(gauss_likelihoods_final_all_obs)):
            file_name = os.path.splitext(os.path.basename(databunch_unlabeled.items[j]))[0]
            file_paths += [databunch_unlabeled.items[j]]
            file_names += [file_name]
            scores += [gauss_likelihoods_final_all_obs[j]]
            labels += [databunch_unlabeled.y[j]]
        # store filtering information
        self.scores = scores
        self.file_paths = file_paths
        self.labels = labels
        self.file_names = file_names

        # copy filtered information to the given folder
        dir_filtered = dir_filtered_root + "/batch_" + str(num_batch_data) + "/batch_" + str(
            num_batch_data) + "_num_unlabeled_" + str(num_unlabeled) + "_ood_perc_" + str(ood_perc) + "/"
        self.copy_filtered_observations(dir_root=dir_filtered, percent_to_filter=ood_thresh)

        print("file_names lenght ", len(file_names))
        print("scores ", len(scores))
        dict_csv = {'File_names': file_names,
                    'Scores': scores, "Info": self.info}
        dataframe = pd.DataFrame(dict_csv, columns=['File_names', 'Scores', 'Info'])
        dataframe.to_csv(path_reports_ood + 'scores_files_batch' + str(num_batch_data) + '.csv', index=False,
                         header=True)
        print(dataframe)



    def copy_filtered_observations(self, dir_root, percent_to_filter):
        """
        Copy filtered observations applying the thresholds
        :param dir_root: directory where to copy the filtered data
        :param   percent_to_filter: percent of observations to keep
        :return:
        """  
        if self.thresholding_method == "kmeans":
            print("using K-Means thresholding")
            thresh = self.kmeans_th(12)
        elif self.thresholding_method == "otsu":
            print("using Otsu thresholding")
            thresh = self.otsu_thresholding(12)
        elif self.thresholding_method == "dbscan":
            print("using DBSCAN thresholding")
            thresh = self.dbscan_thresholding()
        else:
            thresh = self.get_threshold(percent_to_filter)
        
        print("Threshold ", thresh)
        print("Percent to threshold: ", percent_to_filter)
        num_selected = 0
        #store info about the observation filtering
        self.info = [""] * len(self.scores)
        #only filter training
        for i in range(0, len(self.scores)):
            #print("self.file_paths[i]")
            #print(self.file_paths[i])
            #print("Path ", self.file_paths[i])
            #print("Current score ", self.scores[i], " of observation ", i, " condition ", "test" not in str(self.file_paths[i]))
            if(self.scores[i] <= thresh and "test" not in str(self.file_paths[i])):
                num_selected += 1
                rand_class = random.randint(0, self.num_classes - 1)
                path_dest = dir_root + "/train/" + str(rand_class) + "/"
                path_origin = self.file_paths[i]

                try:
                    os.makedirs(path_dest)
                except:
                    a = 0
                file_name = os.path.basename(self.file_paths[i])
                #print("File to copy", path_origin)
                #print("Path to copy", path_dest + file_name)
                shutil.copyfile(path_origin, path_dest + file_name)
                self.info[i] =  "Copied, is training observation lower than thresh " + str(thresh)

            if("test" in str(self.file_paths[i])):
                path_dest = dir_root + "/test/" + str(self.labels[i]) + "/"
                path_origin = self.file_paths[i]
                try:
                    os.makedirs(path_dest)
                except:
                    a = 0
                file_name = os.path.basename(self.file_paths[i])
                #print("File to copy", path_origin)
                #print("Path to copy", path_dest + file_name)
                shutil.copyfile(path_origin, path_dest + file_name)
                self.info[i] = "Copied, is test observation"
        print("Number of unlabeled observations preserved: ", num_selected)

    def get_threshold(self, percent_to_filter):
        """
        Get the threshold according to the list of observations and the percent of data to filter
        :param percent_to_filter: value from 0 to 1
        :return: the threshold
        """
        new_scores_no_validation = []
        for i in range(0, len(self.scores)):
            #bug fixed!!
            if("test" not in str(self.file_paths[i])):
                new_scores_no_validation += [self.scores[i]]

        #percent_to_filter is from  0 to 1
        new_scores_no_validation.sort()
        # condicion si odd perc es 0
        num_to_filter = int(percent_to_filter * len(new_scores_no_validation)) if percent_to_filter < 1 else \
            len(new_scores_no_validation) - 1
        threshold = new_scores_no_validation[num_to_filter]
        return threshold

    def kmeans_th(self, perc):
        new_scores_no_validation = []
        for i in range(0, len(self.scores)):
            #bug fixed!!
            if("test" not in str(self.file_paths[i])):
                new_scores_no_validation += [self.scores[i]]
        X = np.array(new_scores_no_validation)
        m = X.shape[0]
        X = np.reshape(X, [m, 1])
        kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
        
        data = pd.DataFrame()
        data['scores'] = X.ravel()
        data['cluster'] = kmeans.labels_
        
        # Mean and std assuming 1 cluster
        mean_total = np.mean(data["scores"])
        std_total = np.std(data["scores"])
        
        # Mean and std of each cluster
        mean_cluster1, mean_cluster2 = [np.mean(data.loc[np.where(data["cluster"] == k)]["scores"].to_numpy()) for k in range(2)]
        std_cluster1, std_cluster2 = [np.std(data.loc[np.where(data["cluster"] == k)]["scores"].to_numpy()) for k in range(2)]
        
        eps = (perc / 100) * (std_total/mean_total)
        if eps < std_cluster1/mean_cluster1 + std_cluster2/mean_cluster2 - std_total/mean_total:
            clusters = [0] * len(data["scores"])
            T = np.max(data["scores"])
            self.est_clusters_number.append(1)
        else:
            self.est_clusters_number.append(2)
            idx = np.where(data['cluster'] == 0)
            Xi = data.loc[idx]['scores'].to_numpy()
            max1, min1 = (np.max(Xi), np.min(Xi))

            idx = np.where(data['cluster'] == 1)
            Xi = data.loc[idx]['scores'].to_numpy()
            max2, min2 = (np.max(Xi), np.min(Xi))

            T = (np.min([np.abs(min2 - max1), np.abs(min1 - max2)]) / 2) + np.min([max1, max2])
        return T


    def otsu_thresholding(self, perc):
        new_scores_no_validation = []
        print(self.file_paths)
        for i in range(0, len(self.scores)):
            #bug fixed!!
            if("test" not in str(self.file_paths[i])):
                new_scores_no_validation += [self.scores[i]]

        scores = np.array(new_scores_no_validation)
        min_num, max_num = np.min(scores), np.max(scores)
        bins = int((60 / 100) * len(scores))
        h, g = np.histogram(scores, bins=bins, range=[min_num, max_num], density=True)

        h = np.divide(h.ravel(), h.max())
        bin_mids = (g[:-1] + g[1:]) / 2
        weight1 = np.cumsum(h)
        weight2 = np.cumsum(h[::-1])[::-1] 

        mean1 = np.cumsum(h * bin_mids) / weight1
        mean2 = (np.cumsum((h * bin_mids)[::-1]) / weight2[::-1])[::-1]
        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        index_of_max_val = np.argmax(inter_class_variance)
        
        mean_total = np.mean(scores)
        std_total = np.std(scores)
        th = bin_mids[:-1][index_of_max_val]
        m1 = np.mean(scores[scores < th])
        s1 = np.std(scores[scores < th])
        m2 = np.mean(scores[scores >= th])
        s2 = np.std(scores[scores >= th])
        
        eps = (12 / 100) * (std_total / mean_total)
        if eps < s1 / m1 + s2 / m2 - std_total / mean_total:
            print("1 pdf")
            self.est_clusters_number.append(1)
            th = np.max(scores)
        else:
            print("2 pdf")
            self.est_clusters_number.append(2)
        return th
        
        
    
    def get_batch_features(self, tensorbunch_unlabeled, batch_size_unlabeled, batch_number, feature_extractor):
        """
        Get the batcch of features using a specific feature extractor
        :param tensorbunch_unlabeled: tensorbunch to evaluate using the feature extractor
        :param batch_size_unlabeled: batch size to use during evaluation
        :param batch_number: batch number to evaluate
        :param feature_extractor: feature extractor to use
        :return: features extracted
        """

        # create the batch of tensors to get its features
        batch_tensors1 = tensorbunch_unlabeled[
                         batch_number * batch_size_unlabeled:(batch_number + 1) * batch_size_unlabeled, :, :, :]
        # batch indices for accountability
        batch_indices = torch.arange(batch_number * batch_size_unlabeled, (batch_number + 1) * batch_size_unlabeled)
        # print("batch tensors ", batch_tensors1.shape)
        # get the  features from the selected batch
        features_bunch1 = self.extract_features(feature_extractor, batch_tensors1)
        # get the values of a specific dimension
        # values_dimension_bunch1 = features_bunch1[:, :].cpu().detach().numpy()
        values_dimension_bunch1 = features_bunch1[:, :]
        return values_dimension_bunch1, batch_indices


def run_tests_pdf():
    ood_filter_Mahalanobis = OOD_filter_Mahalanobis(model_name = "densenet")

    """
    :param distance: distance_str
    :return:
    """
    #S_l is the IID data for indiana i.e image_67.jpg
    #"/media/Data/saul/Datasets/Covid19/Dataset/batches_labeled_undersampled_in_dist_BINARY_INDIANA_30_val_40_labels"
    #S_u is contaminated dataset
    #/media/Data/saul/Datasets/Covid19/Dataset/OOD_COVID19/OOD_CR_25

    ood_filter_Mahalanobis.run_filter(
        path_bunch1="E:/GoogleDrive/DATASETS_TEMP/batches_labeled_undersampled_in_dist_BINARY_CR_30_val/",
        path_bunch2="E:/GoogleDrive/DATASETS_TEMP/OOD_SIMPLE/batch_", ood_perc=50,
        num_unlabeled=90, num_batches=1, size_image=105, batch_size_p=BATCH_SIZE,
        dir_filtered_root="E:/GoogleDrive/DATASETS_TEMP/Filtered",
        ood_thresh=0.7, path_reports_ood="E:/GoogleDrive/DATASETS_TEMP/Filtered")
