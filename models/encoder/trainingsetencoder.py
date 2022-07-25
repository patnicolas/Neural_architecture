__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import constants
from util.s3util import S3Util
import numpy as np
import os

"""
     Encoding of a given training data set  using the most relevant medical terms, contextual information 
     per each submitted claim.
     The features for each medical term for a given note associated with a submitted claim are 
     - Tf-Idf weight of the most relevant medical terms found in a note
     - Relative, normalized rank of the relevant medical terms in the vocabulary (Indexing)
     - Contextual feature combining age and gender: Male 0.5*age/120 Female 0.5+0.5*age/120
     Grid example
         labeled claim               Term 1               Term 2
         77067 26 Z12.31,R98.1   [0.34, 0.11, 0.89]  [0.18, 0.67, 0.99]  ...

     :param s3_data_source: Name of the data source (i.e. 40/7/utrad)
     :param num_files: Number of files to be used in the encoder
     :param dimension: Model dimension or number of features / 3
 """


class TrainingSetEncoder(object):
    def __init__(self, s3_data_prefix: str, num_files: int, dimension: int):
        s3_bucket = constants.s3_config['bucket_name']
        s3_folder = constants.s3_sources['terms_context_claim']
        s3_data_source = '-'.join([s3_data_prefix, str(dimension)])
        s3_source_folder = os.path.join(s3_folder, s3_data_source)
        s3_data_loader = S3Util(s3_bucket, s3_source_folder, False, num_files, False)
        features = s3_data_loader.s3_to_list(".csv")[:dimension]

        # Generator expression to generate the iterator for the tensor
        def gen(features: list) -> (str, np.array):
            numeric_features = [TrainingSetEncoder.extract_columns(feature) for feature in features if len(feature.split(",")) > 2]
            for numeric_feature in numeric_features:
                first = numeric_feature[0]
                second = numeric_feature[1]
                yield first, second
        self.features_iter = iter(gen(features))

    @staticmethod
    def extract_columns(row: str) -> list:
        """
            Extract the column from a row or entry data loaded from S3
            :param row: Row as a set of values (columns) separated by ','
            :return: List of fields [submitted claim, first relevant medical term, second relevant medical term....]
        """
        raw_fields = row.split(',')
        key = raw_fields[0]
        raw_fields.pop(0)

        # Generator expression to extract the floating values from the raw string field
        def extractor_gen():
            for raw_field in raw_fields:
                yield [float(sub_field) for sub_field in raw_field.replace("\"", "").split('@')]
        return [key, list(extractor_gen())]


    def __repr__(self) -> str:
        return repr(f'Features: relative index, tf-idf, aggregated context\n{self.to_list}')

    def to_list(self) -> list:
        return list(self.features_iter)

    def __len__(self) -> int:
        return len(self.to_list())

    def __next__(self) -> (str, list):
        return next(self.features_iter)

    def to_image(self):
        """
            Convert the value array into an image with 255 as base line
            :return: None
        """
        from PIL import Image
        import matplotlib.pyplot as plt
        import numpy

        # Simple conversion of values into a unsigned int [0, 255]
        # values = [(value*255).astype(numpy.uint8) for _, value in self.to_list()]

        x = [value for _, value in self.to_list()]
        np_x = np.array(x)*255
        img = Image.fromarray(np_x, 'RGB')
        plt.imshow(img)
        img.save('../../images/dsencoder.png')
        plt.show()
