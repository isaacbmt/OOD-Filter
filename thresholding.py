import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import pandas as pd
import shutil
# import time


def normalization(array):
    minimum = np.min(array)
    array_norm = (array - minimum) / (np.max(array) - minimum)
    return array_norm


def plot_histogram(data, th, title, c1, c2, idx, batch):
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

    c1_norm = normalization(norm.pdf(range_plot, c1[0], c1[1])) * np.max(h[0:idx-1])
    plt.fill_between(range_plot, c1_norm, 0, alpha=0.2, color='blue')
    plt.plot(range_plot, c1_norm)

    c2_norm = normalization(norm.pdf(range_plot, c2[0], c2[1])) * np.max(h[idx:-1])
    plt.fill_between(range_plot, c2_norm, 0, alpha=0.2, color='green')
    plt.plot(range_plot, c2_norm)
    plt.show()


def get_reports():
    csv_files = glob.glob(os.path.join('./reports_oodMahalanobis_CHINA65', '*.csv'))
    scores = np.array([])
    for file in csv_files:
        data = pd.read_csv(file)
        scores = np.concatenate((scores, data['Scores']))
    return scores


def get_report(path_reports, batch):
    df = pd.read_csv(f"{path_reports}/scores_files_batch{batch}.csv")
    return df


def kittler(data):
    scores = np.array(data['Scores'])
    min_num, max_num = np.min(scores), np.max(scores)
    bins = round(max_num-min_num)
    h, g = np.histogram(scores,
                           bins=bins,
                           range=[min_num, max_num])
    g = g[:-1]
    C = np.cumsum(h)
    M = np.cumsum(h * g)
    S = np.cumsum(h * g ** 2)
    sigma_f = np.sqrt(S/C - (M/C)**2) + 10**-10

    Cb = (C[-1] - C) + 10**-10
    Mb = M[-1] - M
    Sb = S[-1] - S
    sigma_b = np.sqrt(Sb / Cb - (Mb/Cb)**2) + 10**-10

    P1 = C / C[-1]

    V = P1*np.log(sigma_f) + (1-P1)*np.log(sigma_b) - P1*np.log(P1) - (1-P1)*np.log(1-P1)
    V[~np.isfinite(V)] = np.inf
    idx = np.argmin(V)
    T = g[idx]

    m1 = M[idx] / C[idx]
    sd1 = sigma_f[idx]

    m2 = Mb[idx] / Cb[idx]
    sd2 = sigma_b[idx]

    return T, [m1, sd1], [m2, sd2], idx


def otsu(data):
    scores = np.array(data["Scores"])
    min_num, max_num = np.min(scores), np.max(scores)
    bins = round(max_num - min_num)
    h, g = np.histogram(scores,
                        bins=bins,
                        range=[min_num, max_num])
    bin_mids = (g[:-1] + g [1:]) / 2
    weight1 = np.cumsum(h)
    weight2 = np.cumsum(h[::-1])[::-1] # th = 2133.360345336094

    mean1 = np.cumsum(h * bin_mids) / weight1
    mean2 = (np.cumsum((h * bin_mids)[::-1]) / weight2[::-1])[::-1]

    std1 = np.sqrt(np.cumsum((bin_mids - mean1)**2 * h/weight1))
    std2 = np.sqrt(np.cumsum((bin_mids - mean2)[::-1] ** 2 * (h / weight2)[::-1]))

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    index_of_max_val = np.argmax(inter_class_variance)
    th = bin_mids[:-1][index_of_max_val]

    m1 = mean1[index_of_max_val]
    s1 = std1[index_of_max_val]
    m2 = mean2[::-1][index_of_max_val]
    s2 = std2[index_of_max_val]

    return th, [m1, s1], [m2, s2], index_of_max_val


def kmeans(X, batch, plot):
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

    if plot:
        X = X.ravel()
        plt.scatter(X, np.zeros_like(X) + 0., c=clusters, cmap='rainbow')
        plt.xlabel('Scores - batch ' + str(batch))
        plt.plot(T, 0, color='red', marker='|', linewidth=2, markersize=12)
        plt.show()

    return T


def filter_images(df, th, path_dest, batch, method):
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




def filter_select(method, create, oodperc, plot):
    path_dest = f"dataset/CIFAR10/FILTERED_UNLABELED/{method}/CIFAR{100-oodperc}MNIST{oodperc}"
    # path_reports = f"reportsCIFAR{oodperc}"
    path_reports = "reports_oodMahalanobis_CHINA65"
    T = 500
    for batch in range(1):
        df = get_report(path_reports, batch)
        if create:
            os.makedirs(f"{path_dest}/batch_{batch}")
        if method == "KITTLER":
            T, c1, c2, idx = kittler(df)
            if plot:
                plot_histogram(df['Scores'], T, 'Kittler', c1, c2, idx, batch)
            # print(f'Kittler threshold = {T}')
        elif method == "OTSU":
            T, c1, c2, idx = otsu(df)
            # print(f'Otsu threshold = {T}, c1 = {c1}, c2 = {c2}, idx = {idx}')
            if plot:
                plot_histogram(df['Scores'], T, 'Otsu', c1, c2, idx, batch)
        elif method == "KMEANS":
            T = kmeans(df, batch, plot)
        else:
            print("Method not found")
        # if not plot:
        #     filter_images(df, T, path_dest, batch, method)
    return T


def eval_filter(method, perc, num_unlabeled):
    IOD = []
    OOD = []
    percs = []
    lab_percs = []
    num_unlabeled = 70
    perc = 65
    unlabeled_perc = int(num_unlabeled * perc / 100)
    labeled_perc = int(num_unlabeled * (100 - perc) / 100)

    for batch in range(10):
        path = f'dataset/CIFAR10/FILTERED_UNLABELED/{method}/CIFAR{100-perc}MNIST{perc}/batch_{batch}'
        files = glob.glob(os.path.join(path, '*'))
        ood_files = glob.glob(os.path.join(path, 'ood*'))
        files = pd.read_csv(f"evalChina/{method}/china_filtered_65_OOD_batch_{batch}.csv")
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
        IOD += [len(files) - out]

    dict_csv = {'IOD': IOD, 'OOD': OOD, '%_OOD_filtrado': percs, '%_IOD_filtrado': lab_percs}
    dataframe = pd.DataFrame(dict_csv, columns=['IOD', 'OOD', '%_OOD_filtrado', '%_IOD_filtrado'])
    dataframe.to_csv("eval/" + method + f"_{perc}_OOD" + '.csv', index=False, header=True)

    return [f"China {100-perc}% - CR {perc}",
            f"{int(np.mean(percs))} \u00B1 {int(np.std(percs))}",
            f"{int(np.mean(lab_percs))} \u00B1 {int(np.std(lab_percs))}"]


def get_real_threhold(scores, perc):
    scores = np.array(scores)
    scores.sort()
    print(scores)
    num_to_filter = int(perc * len(scores))
    print(f"perc: {perc} - num: {num_to_filter}")
    if perc == 1:
        threshold = scores[num_to_filter-1]
    else:
        threshold = scores[num_to_filter]
    return threshold


if __name__ == '__main__':
    mode = "real_threshold"
    # percs = [0, 40, 100]
    percs = [65]
    methods = ["KITTLER", "OTSU", "KMEANS"]
    if mode == "filter":
        # methods = ["OTSU"]
        # percs = [40]
        plot = False
        create = False
        used_methods = []
        datasets = []
        thresholds = []
        for perc in percs:
            for method in methods:
                T = filter_select(method, create=create, oodperc=perc, plot=plot)
                used_methods += [method]
                datasets += [f"CHINA {100 - perc}% - CR {perc}%"]
                thresholds += [T]

        dict_csv = {"Metodo": used_methods,
                    "Set de datos": datasets,
                    "Threshold": thresholds}
        dataframe = pd.DataFrame(dict_csv, columns=['Metodo',
                                                    'Set de datos',
                                                    'Threshold'])
        dataframe.to_csv(f"report_threshold/CHINA_methods.csv", index=False, header=True)

    elif mode == "eval":
        for method in methods:
            dataset = []
            unlabeled_perc = []
            labeled_perc = []
            for perc in percs:
                info = eval_filter(method, perc, 90)
                dataset += [info[0]]
                unlabeled_perc += [info[1]]
                labeled_perc += [info[2]]
            dict_csv = {"Set de datos": dataset,
                        "% filtrado observaciones OOD": unlabeled_perc,
                        "% filtrado observaciones IOD": labeled_perc}
            dataframe = pd.DataFrame(dict_csv, columns=['Set de datos',
                                                        '% filtrado observaciones OOD',
                                                        '% filtrado observaciones IOD'])
            dataframe.to_csv("evalChina/" + method + "_result" + '.csv', index=False, header=True)
    elif mode == "real_threshold":
        report_path = "report_threshold/"
        datasets = []
        thresholds = []
        for perc in percs:
            path_reports = f"reports_oodMahalanobis_CHINA65"
            df = get_report(path_reports, 0)
            T = get_real_threhold(df["Scores"], (100 - perc) / 100)
            datasets += [f"CHINA {100-perc}% - CR {perc}%"]
            thresholds += [T]
        dict_csv = {"Set de datos": datasets,
                    "Threshold real": thresholds}
        dataframe = pd.DataFrame(dict_csv, columns=['Set de datos',
                                                    'Threshold real'])
        dataframe.to_csv(f"report_threshold/CHINA.csv", index=False, header=True)



