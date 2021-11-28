import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans
import pandas as pd
import shutil


class Thresholding:
    def __init__(self, reports_path="", dest_path="", eval_path="", dataset1_name="", dataset2_name="",
                 num_unlabeled=100, ood_perc=100, save_files=False,
                 plot=False, create_batch_dir=False, num_batches=10):
        """
        :param reports_path: Path where reports are located
        :param dest_path: Path where results are going to be saved
        :param eval_path: Path inside <dest_path> where evaluation results are going to be saved
        :param dataset1_name: ID dataset name
        :param dataset2_name: OOD dataset name
        :param num_unlabeled: Number of unlabeled samples
        :param ood_perc: Percentage of OOD samples (used for evaluation purposes)
        :param save_files: Flag to define if filtered files should be saved
        :param plot: Graph the data annd their threshold
        :param create_batch_dir: Create directory to save the filtered files
        :param num_batches: Number of batches
        """
        self.reports_path = reports_path
        self.dest_path = dest_path
        self.eval_path = eval_path
        self.dataset1_name = dataset1_name
        self.dataset2_name = dataset2_name
        self.num_unlabeled = num_unlabeled
        self.ood_perc = ood_perc
        self.save_files = save_files
        self.plot = plot
        self.create_batch_dir = create_batch_dir
        self.num_batches = num_batches

    def normalization(self, array):
        minimum = np.min(array)
        array_norm = (array - minimum) / (np.max(array) - minimum)
        return array_norm

    def plot_histogram(self, data, th, title, c1, c2, idx, batch):
        """

        :param data:
        :param th: Threshold
        :param title:
        :param c1: curve 1 is a list that contains [mean, standard deviation]
        :param c2: curve 2 is a list that contains [mean, standard deviation]
        :param idx: index where threshold is located in data
        :param batch: number of batch that is being plotted
        """
        plt.hist(data, bins=50)
        scores = data
        min_num, max_num = np.min(scores), np.max(scores)
        bins = round(max_num - min_num)
        # bins =30
        h, g = np.histogram(scores,
                            bins=bins,
                            range=[min_num, max_num])
        g = g[:-1]
        plt.xlabel('Scores - batch ' + str(batch))
        plt.ylabel('h(g)')
        plt.title(title)
        plt.plot(th, 0, color='red', marker='|', linewidth=2, markersize=12)
        # plt.plot(g, h)

        range_plot = np.arange(np.min(data), np.max(data), 0.01)

        c1_norm = self.normalization(norm.pdf(range_plot, c1[0], c1[1])) * np.max(h[0:idx - 1])
        plt.fill_between(range_plot, c1_norm, 0, alpha=0.2, color='blue')
        plt.plot(range_plot, c1_norm)

        c2_norm = self.normalization(norm.pdf(range_plot, c2[0], c2[1])) * np.max(h[idx:-1])
        plt.fill_between(range_plot, c2_norm, 0, alpha=0.2, color='green')
        plt.plot(range_plot, c2_norm)
        plt.show()

    def get_reports(self):
        csv_files = glob.glob(os.path.join('./reports_oodMahalanobis_CHINA65', '*.csv'))
        scores = np.array([])
        for file in csv_files:
            data = pd.read_csv(file)
            scores = np.concatenate((scores, data['Scores']))
        return scores

    def get_report(self, path_reports, batch):
        df = pd.read_csv(f"{path_reports}/scores_files_batch{batch}.csv")
        return df

    def kittler(self, data):
        """
        Obtains the threshold using the minimum thresholding of Kittler
        :param data: Array of numbers
        :return: Threshold, curve1: [mean, std], curve2: [mean, std], index of threshold
        """
        scores = np.array(data['Scores'])
        min_num, max_num = np.min(scores), np.max(scores)
        bins = round(max_num - min_num)
        h, g = np.histogram(scores,
                            bins=bins,
                            range=[min_num, max_num])
        g = g[:-1]
        C = np.cumsum(h)
        M = np.cumsum(h * g)
        S = np.cumsum(h * g ** 2)
        sigma_f = np.sqrt(S / C - (M / C) ** 2) + 10 ** -10

        Cb = (C[-1] - C) + 10 ** -10
        Mb = M[-1] - M
        Sb = S[-1] - S
        sigma_b = np.sqrt(Sb / Cb - (Mb / Cb) ** 2) + 10 ** -10

        P1 = C / C[-1]

        V = P1 * np.log(sigma_f) + (1 - P1) * np.log(sigma_b) - P1 * np.log(P1) - (1 - P1) * np.log(1 - P1)
        V[~np.isfinite(V)] = np.inf
        idx = np.argmin(V)
        T = g[idx]

        m1 = M[idx] / C[idx]
        sd1 = sigma_f[idx]

        m2 = Mb[idx] / Cb[idx]
        sd2 = sigma_b[idx]

        return T, [m1, sd1], [m2, sd2], idx

    def otsu(self, data):
        """
        Obtains the threshold using the thresholding of Otsu
        :param data: Array of numbers
        :return: Threshold, curve1: [mean, std], curve2: [mean, std], index of threshold
        """
        scores = np.array(data["Scores"])
        min_num, max_num = np.min(scores), np.max(scores)
        bins = round(max_num - min_num)
        h, g = np.histogram(scores,
                            bins=bins,
                            range=[min_num, max_num])
        bin_mids = (g[:-1] + g[1:]) / 2
        weight1 = np.cumsum(h)
        weight2 = np.cumsum(h[::-1])[::-1]  # th = 2133.360345336094

        mean1 = np.cumsum(h * bin_mids) / weight1
        mean2 = (np.cumsum((h * bin_mids)[::-1]) / weight2[::-1])[::-1]

        std1 = np.sqrt(np.cumsum((bin_mids - mean1) ** 2 * h / weight1))
        std2 = np.sqrt(np.cumsum((bin_mids - mean2)[::-1] ** 2 * (h / weight2)[::-1]))

        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        index_of_max_val = np.argmax(inter_class_variance)
        th = bin_mids[:-1][index_of_max_val]

        m1 = mean1[index_of_max_val]
        s1 = std1[index_of_max_val]
        m2 = mean2[::-1][index_of_max_val]
        s2 = std2[index_of_max_val]

        return th, [m1, s1], [m2, s2], index_of_max_val

    def kmeans(self, X, batch):
        """
        Obtains theshold using a modified K-means algorithm for dynamic clustering
        :param X: data
        :param batch: number of batch which the data belongs
        :return: Threshold
        """
        X = np.array(X["Scores"])
        m = X.shape[0]
        X = np.reshape(X, [m, 1])
        k = 2
        inter = -float('inf')
        intra = float('inf')
        data = pd.DataFrame()
        while k <= 2:
            kmeans = KMeans(n_clusters=k, random_state=10).fit(X)
            centers = kmeans.cluster_centers_
            clusters = kmeans.labels_

            # (scores, columns=['scores'])
            data['scores'] = X.ravel()
            data['cluster'] = clusters

            new_inter = np.min(centers[0:k - 1] - centers[1:k])
            new_intra = 0
            for c in range(k):
                idx = np.where(data['cluster'] == c)
                Xi = data.loc[idx]['scores'].to_numpy()
                Xm = Xi.mean()
                n = Xi.shape[0]
                new_intra += np.sqrt(1 / (n - 1)) * np.sum((Xi - Xm) ** 2)

            if new_intra < intra and new_inter > inter:
                intra = new_intra
                inter = new_inter
                k += 1
            else:
                break

        idx = np.where(data['cluster'] == 0)
        Xi = data.loc[idx]['scores'].to_numpy()
        max1, min1 = (np.max(Xi), np.min(Xi))

        idx = np.where(data['cluster'] == 1)
        Xi = data.loc[idx]['scores'].to_numpy()
        max2, min2 = (np.max(Xi), np.min(Xi))

        T = (np.min([np.abs(min2 - max1), np.abs(min1 - max2)]) / 2) + np.min([max1, max2])

        if self.plot:
            X = X.ravel()
            plt.scatter(X, np.zeros_like(X) + 0., c=clusters, cmap='rainbow')
            plt.xlabel('Scores - batch ' + str(batch))
            plt.plot(T, 0, color='red', marker='|', linewidth=2, markersize=12)
            plt.show()

        return T

    def filter_images(self, df, th, path_dest, batch, method):
        """
        Save the images with a score below the given threshold in the destination path
        :param df: Dataframe
        :param th: Threshold
        :param path_dest: Destination path to save the files
        :param batch: Number of batch
        :param method: Name of method used
        """
        for index, row in df.iterrows():
            if row["Scores"] <= th:
                file_path = row['File_paths']
                filename = os.path.basename(file_path)
                dest = f"{path_dest}/batch_{batch}/{filename}"
                shutil.copyfile(file_path, dest)

        # for index, row in df.iterrows():
        #     if row["Scores"] <= th:
        #         file_names += [row["File_names"]]
        #         scores += [row["Scores"]]
        #
        # dict_csv = {'File_names': file_names,
        #             'Scores': scores}
        # dataframe = pd.DataFrame(dict_csv, columns=['File_names', 'Scores'])
        # dataframe.to_csv(f"evalChina/{method}/china_filtered_{65}_OOD_batch_{batch}.csv", index=False, header=True)

    def filter_select(self, method, oodperc):
        """
        Filter the files using a given method
        :param method: Name of method
        :param oodperc: OOD percentage
        :return: Thesholds
        """
        path_dest = f"{self.dest_path}/{self.dataset1_name}/FILTERED_UNLABELED/{method}/{self.dataset1_name}{100 - oodperc}_{self.dataset2_name}{oodperc}"
        T = 500
        T_array = []
        for batch in range(self.num_batches):
            print(f"Batch num: {batch}")
            df = self.get_report(self.reports_path, batch)
            if self.create_batch_dir:
                os.makedirs(f"{path_dest}/batch_{batch}")
            if method == "KITTLER":
                T, c1, c2, idx = self.kittler(df)
                if self.plot:
                    self.plot_histogram(df['Scores'], T, 'Kittler', c1, c2, idx, batch)
                # print(f'Kittler threshold = {T}')
            elif method == "OTSU":
                T, c1, c2, idx = self.otsu(df)
                # print(f'Otsu threshold = {T}, c1 = {c1}, c2 = {c2}, idx = {idx}')
                if self.plot:
                    self.plot_histogram(df['Scores'], T, 'Otsu', c1, c2, idx, batch)
            elif method == "KMEANS":
                T = self.kmeans(df, batch)
            else:
                print("Method not found")
            if self.save_files:
                self.filter_images(df, T, path_dest, batch, method)

            T_array += [T]
        return T_array

    def run_filter(self):
        methods = ["KITTLER", "OTSU", "KMEANS"]
        used_methods = []
        # datasets = []
        thresholds = []
        # for perc in self.percentages:
        for method in methods:
            print(f"Running: {method}")
            T = self.filter_select(method, oodperc=self.ood_perc)
            used_methods += [method] * self.num_batches
            thresholds += T

        total_batches = np.arange(self.num_batches).tolist() * 3
        dict_csv = {"Metodo": used_methods, 'Batch': total_batches, "Threshold": thresholds}
        dataframe = pd.DataFrame(dict_csv, columns=['Metodo', "Batch", 'Threshold'])
        csv_path = f"{self.dest_path}/reports/threshold_{self.dataset1_name}_ood{self.ood_perc}.csv"
        print(csv_path)
        dataframe.to_csv(csv_path, index=False, header=True)

    def run_eval(self):
        methods = ["KITTLER", "OTSU", "KMEANS"]
        dataset = []
        unlabeled_perc = []
        labeled_perc = []
        for method in methods:
            info = self.eval_filter(method, self.ood_perc, self.num_unlabeled)
            dataset += [info[0]]
            unlabeled_perc += [info[1]]
            labeled_perc += [info[2]]
        dict_csv = {"Set de datos": dataset,
                    "% filtrado observaciones OOD": unlabeled_perc,
                    "% filtrado observaciones ID": labeled_perc}
        dataframe = pd.DataFrame(dict_csv, columns=['Set de datos',
                                                    '% filtrado observaciones OOD',
                                                    '% filtrado observaciones ID'])
        dataframe.to_csv(f"{self.dest_path}/{self.eval_path}/" + method + "_result" + '.csv', index=False, header=True)

    def eval_filter(self, method, perc, num_unlabeled):
        ID = []
        OOD = []
        percs = []
        lab_percs = []
        unlabeled_perc = int(num_unlabeled * perc / 100)
        labeled_perc = int(num_unlabeled * (100 - perc) / 100)

        for batch in range(self.num_batches):
            # path = f'dataset/CIFAR10/FILTERED_UNLABELED/{method}/{self.dataset1_name}{100 - perc}{self.dataset2_name}{perc}/batch_{batch}'
            path_dest = f"{self.dest_path}/{self.dataset1_name}/FILTERED_UNLABELED/{method}/{self.dataset1_name}{100 - perc}_{self.dataset2_name}{perc}/batch_{batch}"
            files = glob.glob(os.path.join(path_dest, '*'))
            # files = pd.read_csv(f"evalChina/{method}/china_filtered_65_OOD_batch_{batch}.csv")
            ood_files = []
            for uid, row in files.iterrows():
                if row["File_names"][0:3] == "ood":
                    ood_files += [row["File_names"]]

            out = len(ood_files)
            if labeled_perc != 0:
                lab_percs += [(labeled_perc - (len(files) - out)) * 100 / labeled_perc]
            else:
                lab_percs += [0]

            if unlabeled_perc != 0:
                percs += [(unlabeled_perc - out) * 100 / unlabeled_perc]
            else:
                percs += [0]
            OOD += [out]
            ID += [len(files) - out]

        # dict_csv = {'ID': ID, 'OOD': OOD, '%_OOD_filtrado': percs, '%_ID_filtrado': lab_percs}
        # dataframe = pd.DataFrame(dict_csv, columns=['ID', 'OOD', '%_OOD_filtrado', '%_ID_filtrado'])
        # dataframe.to_csv(f"{self.dest_path}/evaluation/{method}/" + f"percentages_OOD_{perc}" + '.csv', index=False, header=True)

        return [f"{self.dataset1_name} {100 - perc}% - {self.dataset1_name} {perc}",
                f"{int(np.mean(percs))} \u00B1 {int(np.std(percs))}",
                f"{int(np.mean(lab_percs))} \u00B1 {int(np.std(lab_percs))}"]

    def get_real_threhold(self, scores, perc):
        scores = np.array(scores)
        scores.sort()
        print(scores)
        num_to_filter = int(perc * len(scores))
        print(f"perc: {perc} - num: {num_to_filter}")
        if perc == 1:
            threshold = scores[num_to_filter - 1]
        else:
            threshold = scores[num_to_filter]
        return threshold


