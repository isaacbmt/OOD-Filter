# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from datetime import datetime
from collections import defaultdict
import threading
import time
import logging
import os
import sys
import pandas as pd
from pandas import DataFrame
from collections import defaultdict
from csv import writer

class TrainLog:
    """Saves training logs in Pandas csvs"""

    INCREMENTAL_UPDATE_TIME = 300

    def __init__(self, directory, name):
        """
        Constructor
        :param directory: directory of the trainlog
        :param name: name of the trainlog
        """
        self.log_file_path = "{}/{}.csv".format(directory, name)
        self._log = defaultdict(dict)
        self._log_lock = threading.RLock()
        self._last_update_time = time.time() - self.INCREMENTAL_UPDATE_TIME

    def record_single(self, step, column, value):
        """
        Single recording
        :param step: row
        :param column: column
        :param value: value to store
        :return:
        """
        self._record(step, {column: value})

    def record(self, step, col_val_dict):
        """
        Record in a specific row, dictionary entry
        :param step:
        :param col_val_dict:
        :return:
        """
        self._record(step, col_val_dict)

    def save(self):
        """
        Save the trainlog
        :return:
        """
        df = self._as_dataframe()
        #df.to_msgpack(self.log_file_path, compress='zlib')
        df.to_csv(self.log_file_path)

    def _record(self, step, col_val_dict):
        """
        Record with time
        :param step:
        :param col_val_dict:
        :return:
        """
        with self._log_lock:
            self._log[step].update(col_val_dict)
            #if time.time() - self._last_update_time >= self.INCREMENTAL_UPDATE_TIME:
            self._last_update_time = time.time()
            self.save()

    def _as_dataframe(self):
        """
        Transform to dataframe
        :return:
        """
        with self._log_lock:
            return DataFrame.from_dict(self._log, orient='index')


class RunContext:
    """Creates directories and files for the run"""
    def __init__(self, logging, args):
        """
        Inits the run context
        @param logging: logger
        @param args: received and parsed arguments
        """
        name_log_folder = args.log_folder
        self.dateInfo = "{date:%Y-%m-%d_%H_%M_%S}".format(date=datetime.now())
        #runner_name = os.path.basename(runner_file).split(".")[0]
        self.result_dir = ("../../{root}/" + self.dateInfo + "/").format(root = name_log_folder)
        #transient dir contains the checkpoints, information logs and training logs
        self.transient_dir = self.result_dir + "/logs/"
        os.makedirs(self.result_dir)
        os.makedirs(self.transient_dir)
        logging.basicConfig(filename=self.transient_dir + "log_" + self.dateInfo + ".txt", level=logging.INFO, format='%(message)s')
        self.LOG = logging.getLogger('main')
        self.init_logger()
        #creating log in log dir
        self.LOG.info("Creating directories for execution: ")
        self.LOG.info(self.result_dir)
        self.LOG.info(self.transient_dir)
        self.write_args_log(args)


    def write_args_log(self, args):
        """"
        Verbose to file of the received arguments
        @param args: received arguments
        """
        self.LOG.info("List of parameters")
        self.LOG.info(str(args))


    def write_run_log(self, run_log_pandas, name_results_log, F1_score, precision, recall):
        """
        Write the given log
        :param run_log_pandas: pandas log to write
        :param name_results_log:  name of the log
        :param F1_score: f1 score to report
        :param precision: precision to report
        :param recall: recall to report
        :return:
        """
        name_run_log = self.transient_dir + "run_log_" + self.dateInfo + ".csv"
        self.LOG.info("Writing run log to : " + name_run_log)
        run_log_pandas.to_csv(name_run_log)
        maximum_validation_acc = run_log_pandas['accuracy'].max()
        minimum_train_loss = run_log_pandas['train_loss'].min()
        self.LOG.info("Maximum accuracy yielded: " + str(maximum_validation_acc))
        self.LOG.info("Minimum training loss: " + str(minimum_train_loss))
        name_results_log = "summaries/" + name_results_log
        new_row = [name_run_log, minimum_train_loss, maximum_validation_acc, F1_score, precision, recall]
        with open(name_results_log, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(new_row)
            self.LOG.info("Stats file written in: " + name_results_log)
            write_obj.close()

    def write_run_log_all_test_metrics_fastai(self, run_log_pandas, name_results_log, path_data, list_metrics,
                                              list_metrics_names):
        """
        Write all metrics to log
        :param run_log_pandas: pandas log
        :param name_results_log: name of the log
        :param path_data: path to the data used
        :param list_metrics: list of the metrics to log
        :param list_metrics_names: names list of the metrics to log
        :return:
        """
        name_run_log = self.transient_dir + "run_log_" + self.dateInfo + ".csv"
        self.LOG.info("Writing run log to : " + name_run_log)
        run_log_pandas.to_csv(name_run_log)
        #print("run_log_pandas")
        #print(run_log_pandas)
        # get index of max val accuracy
        maximum_validation_acc = run_log_pandas['accuracy'].max()
        minimum_train_loss = run_log_pandas['train_loss'].min()
        self.LOG.info("Maximum accuracy yielded: " + str(maximum_validation_acc))

        self.LOG.info("Minimum training loss: " + str(minimum_train_loss))
        name_results_log = r"summaries/" + name_results_log
        new_row = [name_run_log, minimum_train_loss, maximum_validation_acc] + list_metrics
        with open(name_results_log, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            # if its the first batch, write the table header
            if ("batch_0" in path_data):
                header_row = ["Log_name", "Min_train_loss", "maximum_validation_acc"] + list_metrics_names
                csv_writer.writerow(header_row)
            csv_writer.writerow(new_row)
            self.LOG.info("Stats file written in: " + name_results_log)
            write_obj.close()

    def write_run_log_precision_recall_fastai(self, run_log_pandas, name_results_log, path_data, list_metrics, list_metrics_names):
        """
        Write the precision recall info of validation for the best model got, in the learner
        :param run_log_pandas:  pandas log
        :param name_results_log: name of the log
        :param path_data: path to the data
        :param list_metrics: list of the metrics
        :param list_metrics_names: metrics names
        :return:
        """
        name_run_log = self.transient_dir + "run_log_" + self.dateInfo + ".csv"
        self.LOG.info("Writing run log to : " + name_run_log)
        run_log_pandas.to_csv(name_run_log)

        print("run_log_pandas")
        print(run_log_pandas)
        #get index of max val accuracy
        maximum_validation_acc = run_log_pandas['accuracy'].max()
        index_max_acc = run_log_pandas.index[run_log_pandas['accuracy'] == maximum_validation_acc]
        index_max_acc = index_max_acc[0]
        print("index_max_acc")
        print(index_max_acc)


        #get index of recall and precision, based on the maximum accuracy
        maximum_validation_recall = run_log_pandas['recall'][index_max_acc]
        maximum_validation_precision = run_log_pandas['precision'][index_max_acc]
        print("maximum_validation_recall before to list ", maximum_validation_recall)
        print("maximum_validation_precision before to list ", maximum_validation_precision)
        #maximum_validation_recall = maximum_validation_recall[0].tolist()
        #maximum_validation_precision = maximum_validation_precision[0].tolist()

        print("type maximum_validation_precision ", type(maximum_validation_precision))
        print("type maximum_validation_recall ", type(maximum_validation_recall))
        minimum_train_loss = run_log_pandas['train_loss'].min()
        self.LOG.info("Maximum accuracy yielded: " + str(maximum_validation_acc))
        self.LOG.info("Its recall: " + str(maximum_validation_recall))
        self.LOG.info("Its precision: " + str(maximum_validation_precision))
        self.LOG.info("Minimum training loss: " + str(minimum_train_loss))
        name_results_log = "summaries/" + name_results_log
        new_row = [name_run_log, minimum_train_loss, maximum_validation_acc, maximum_validation_recall, maximum_validation_precision] + list_metrics
        with open(name_results_log, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            #if its the first batch, write the table header
            if("batch_0" in path_data):
                header_row = ["Log_name", "Min_train_loss", "maximum_validation_acc", "maximum_validation_recall", "maximum_validation_precision"] + list_metrics_names
                csv_writer.writerow(header_row)
            csv_writer.writerow(new_row)
            self.LOG.info("Stats file written in: " + name_results_log)
            write_obj.close()

    def init_logger(self):
        """
        Sets logging details
        :return:
        """
        #self.LOG.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.LOG.addHandler(handler)

    def get_logger(self):
        """
        Get the logger
        :return:
        """
        return self.LOG

    def create_train_log(self, name):
        """
        Create train logger
        :param name:
        :return:
        """
        return TrainLog(self.transient_dir, name)

    def create_results_all_log(self,name, directory = "../logs/summary"):
        """
        Create Results to the log
        :param name:
        :param directory:
        :return:
        """
        return TrainLog(directory, name)
