__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import constants
from torch.utils.data import Dataset
from wordembedding.context import Context
from models.encoder.vocabulary import Vocabulary
import pandas as pd
from datasets.s3datasetloader import S3DatasetLoader


def split(s: str) -> list:
    return split(s)

"""
    Dataset for NGram extracted from a note or document given a context. The extraction is performed against
    a vocabulary of keywords then potentially processed through a context.
    A context assembles keywords into NGrams (i.e. word2vec) for further processing. If the context is not defined
    in the constructor, the NGram are the orginal keywords
        {Document} - [Parser] -> {keywords} -> [Context] -> {NGrams}
    The data set can be instantiated from
    - Pandas data frame  (__init__)
    - Local JSON file  (from_json_file)
    - S3/HDFS files  (from_s3)
    
    :param df: DataFrame
    :param context: Optional wrapper for the contextual information (Query, Word2Vec,...)
"""


class NGramDataset(Dataset):
    def __init__(self, df: pd.DataFrame, context: Context = None):
        super(NGramDataset, self).__init__()
        if context is not None:
            target_df = context.apply_lst(df.values)
        else:
            target_df = df.apply(func = lambda s: split(s), axis=1)

        vocabulary = Vocabulary()
        # Build the sequence of input_tensors
        self.target_tensors = []
        for ctx_tgt_pairs in target_df:
            tensor_pairs = [(torch.tensor([vocabulary.word_index_dict[c] for c in ctx], device=constants.torch_device), \
                             torch.tensor(vocabulary.word_index_dict[tgt], device=constants.torch_device)) for ctx, tgt in ctx_tgt_pairs]
            for tensor_pair in tensor_pairs:
                self.target_tensors.append(tensor_pair)


    @classmethod
    def from_json_files(cls,
                        input_path: str,
                        context: Context,
                        feature_names: list,
                        label_name: str,
                        limit: int = -1) -> Dataset:
        """
            Alternative constructor using the content of a file as input_tensor. (Decorator design pattern)
            :param input_path: Path of file containing input_tensor data
            :param context: Contextual data
            :param feature_names:  Optional list of feature_name to extract numerical data
            :param label_name: Optional label key
            :param limit: Number of data points (all data points if -1) to be processed
            :return: A data set of type NGramDataset
        """
        df = pd.read_json(input_path)
        num_rows = len(list(df))
        if limit == -1 and num_rows < limit:
            return cls(df[[*feature_names, label_name]], context)
        else:
            return cls(df[[*feature_names, label_name]][0:limit], context)


    '''
        Alternative constructor using the content of a  S3 file as input_tensor. (Decorator design pattern)
        :param s3_bucket: S3 bucket containing the input_tensor data
        :param s3_folder: Path of file containing input_tensor data
        :param context: Contextual onformation 
        :param text_col_name: Optional name of columns for encoder and label
        :param file_extension: File extension used to filter the files from which the encoder and label are extracted
        :returns: Dataset
    '''
    @classmethod
    def from_s3(cls,
                s3_bucket: str,
                s3_folder: str,
                context: Context,
                col_names: list,
                file_extension: str,
                limit: int = -1) -> Dataset:
        dataset_s3_loader = S3DatasetLoader(s3_bucket, s3_folder, col_names, False, file_extension)
        # Extract the data frame related to the column names
        if len(col_names) == 1:
            df = dataset_s3_loader.df[0]
        else:
            df = dataset_s3_loader.df[0:len(col_names)]
        # Apply num_samples if specify
        if limit < 1 and len(df) < limit:
            return cls(df, context)
        else:
            return cls(df[0:limit], context)


    def __len__(self) -> int:
        return len(self.target_tensors)

    def __getitem__(self, item: int) -> (torch.Tensor, torch.Tensor):
        try:
            return self.target_tensors[item]
        except IndexError as e:
            constants.log_error(f'Index {item} out of range {str(e)}')
            return None

    def __repr__(self):
        return repr(self.target_tensors)

    @staticmethod
    def collate_batch(batch: list) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        contexts = [ctx.tolist() for ctx, _ in batch]
        context_size = len(contexts[0])
        xs = []
        for lst in contexts:
            for x in lst:
                xs.append(x)
        offsets = [offset for offset in range(0, len(xs), context_size)]

        # contexts = lambda t: [row for sublist in t for row in sublist]
        labels = [tgt.tolist() for _, tgt in batch]
        return torch.tensor(xs), torch.tensor(labels), torch.tensor(offsets[:-1])


