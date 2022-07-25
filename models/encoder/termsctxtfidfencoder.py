__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import constants
import os
from util.s3util import S3Util
import numpy
import itertools
import torch
import json
from types import SimpleNamespace

"""
     Create an encoded training set for terms context tf-idf representation
     :param s3_data_source:  S3 data source
     :param num_files: Number of files to be loaded from S3 directory
 """

class TermsCtxTfIdfEncoder(object):
    def __init__(self, s3_data_source: str, num_files: int, sample_size: int = -1):
        s3_bucket = constants.s3_config['bucket_name']
        s3_folder = constants.s3_sources['terms_features']
        self.dimension = 0
        # oad and create an indexed map for target or labels
        self.__create_label_indices(s3_bucket, s3_folder, s3_data_source)

        s3_data_folder = os.path.join(s3_folder, s3_data_source)
        s3_data_loader = S3Util(s3_bucket, s3_data_folder, False, num_files, False)
        training_data_list = s3_data_loader.s3_to_list(".json")

        # We use an iterator for the raw data
        label_features_iter = (json.loads(training_data, object_hook=lambda x: SimpleNamespace(**x))
                                    for training_data in training_data_list)
        # Filter the iterator using itertools
        self.label_features_iter = itertools.filterfalse(lambda x: x.label not in self.label_index_map, label_features_iter)

        """
        def weight_feature(label: str, record_size: int, fields: list, indices: list) -> (str, numpy.array):
            features = TermsCtxTfIdfEncoder.to_dense(fields, indices, record_size)
            if self.dimension == 0:
                self.dimension = len(features)
            features[len(features)-1] = features[len(features)-1]*30
            return (self.label_index_map[label], features)

        def gen_labels_features(training_data_iter):
            running = True
            while running:
                try:
                    records_batch = next(training_data_iter)
                    for record in records_batch:
                        r = json.loads(record, object_hook=lambda x: SimpleNamespace(**x))
                        if r.label in self.label_index_map:
                            yield weight_feature(r.label, r.size, r.fields, r.indices)
                except StopIteration as e:
                    constants.log_info(str(e))
                    running = False
        self.label_features_iter = gen_labels_features(training_data_iter)
        # DEBUG
        a = self.label_features_iter
        print(next(a))
        """



    def __create_label_indices(self, s3_bucket: str, s3_folder: str, s3_data_source: str) -> (dict, dict):
        s3_indexed_folder = os.path.join(s3_folder, "index", s3_data_source)
        s3_data_loader = S3Util(s3_bucket, s3_indexed_folder, False, 10, False)

        def extract_indexed_label(indexed_data: str):
            index_str = indexed_data.split(',')
            index = int(index_str[1].replace('\n', ''))
            return index_str[0], index

        indexed_labels_list = list(s3_data_loader.s3_file_to_iter())
        label_index_pairs = [extract_indexed_label(indexed_label) for indexed_label in indexed_labels_list]
        index_label_pairs = [(index, label) for label, index in label_index_pairs ]
        self.label_index_map = dict(label_index_pairs)
        self.index_label_map = dict(index_label_pairs)
        self.count = len(self.index_label_map)

        del indexed_labels_list, label_index_pairs, index_label_pairs


    def next(self) -> (str, numpy.array):
        return next(self.label_features_iter)

    def to_list(self) -> list:
        """
            Convert the iterator to a list of data (label, features)
            :return: List of pair (label, features)
        """
        return list(self.label_features_iter)

