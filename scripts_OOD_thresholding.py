from OOD_thresholding import Thresholding


def run_OOD_thresholding_CIFAR():
    """
    Instructions to run:
    - In order to run the filter, the class Thresholding needs to following attributes:
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

    - To filter the images using the thresholding methods of:
        * Otsu
        * Kittler
        * Dynamic clustering of K-means
      The method <run_filter> is used.

    - The method <run_eval> is used to obtain the percentage of data OOD filtered.
    """
    ood_filter = Thresholding(reports_path="reportsCIFAR0", dest_path="results", eval_path="evaluation",
                              dataset1_name="CIFAR", dataset2_name="MNIST",
                              num_unlabeled=90, ood_perc=0, save_files=False,
                              plot=False, create_batch_dir=True, num_batches=10)
    ood_filter.run_filter()
    ood_filter.run_eval()


run_OOD_thresholding_CIFAR()
