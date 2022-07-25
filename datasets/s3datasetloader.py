__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from util.s3util import S3Util
import constants
import pandas as pd

"""
    Helper/utility class to load a Pandas data frame from S3 folder
    :param s3_bucket: Bucket on S3
    :param s3_folder: Folder containing the dataset data
    :param feature_names: Name of column_names from which to extract data
    :param is_nested If the field under text_col_name is nested into a higher level column (i.e. list, dictionary, ...)
    :param filter_file_extension: File extension potentially used to filter the files in the folder. All files are
        loaded if the extension filter is None
    :param num_samples: Number of samples or documents used by the data loader (the entire dataset if -1)
    
    This class relies on the S3Util class
"""


class S3DatasetLoader(object):
    def __init__(self,
                 s3_bucket: str,
                 s3_folder: str,
                 col_names: list,
                 is_nested: bool,
                 filter_file_extension: str,
                 num_samples: int = -1):
        s3_util = S3Util(s3_bucket, s3_folder, is_nested, num_samples)
        self.df = s3_util.s3_to_dataframe(filter_file_extension, col_names)

    def __repr__(self) -> str:
        return repr(self.df)

    def __getitem__(self, key: str) -> pd.DataFrame:
        """
            Implement the subscript operator [] for this data set.
            :param key: Key or column name
            :returns: A single column data frame
        """
        try:
            self.df[key]
        except KeyError as e:
            constants.log_error(str(e))


