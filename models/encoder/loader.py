__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."


from util.s3util import S3Util
from util.logger import Logger

class Loader(object):
    def __init__(self, folder, s3_bucket_name = ""):
        if s3_bucket_name != "":
            s3_util = S3Util(s3_bucket_name, folder, True)
            self.df = s3_util.to_dataframe()
        else:
            logging = Logger(folder)
            self.df = logging.to_dataframe()
