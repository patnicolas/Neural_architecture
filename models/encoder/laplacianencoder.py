__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from models.encoder.featureencoder import FeatureEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import constants
from util.ioutil import IOUtil
import pandas as pd
import numpy as np

"""
    Implementation of the Laplacian tensor D - A with the following features
    - Laplacian value Normalized Degree - Adjacency values
    - Normalized index of the doc_terms_str in adjacency matrix column
    - Normalized index of the doc_terms_str in adjacency matrix row
    - TfIdf values
    
    :param text_col_name: Target or name of column to extract doc_terms_str
    :param encoding_scheme: Encoding scheme
    :param vocabulary: Vocabulary
    :param max_num_items: Maximum number of doc_terms_str to be used in Laplacian
"""


class LaplacianEncoder(FeatureEncoder):
    def __init__(self, text_col_name: str, term_index_weights: dict, max_num_items: int):
        super(LaplacianEncoder, self).__init__(text_col_name)
        self.term_index_weights = term_index_weights
        self.max_num_terms = max_num_items

    def encode(self, feature_df: pd.DataFrame, stored: bool = None) -> np.array:
        """
            Encode features extracted from a Pandas data frame
            :param feature_df: Data frame containing the words defined in 'text_col_name' column
            :return: Numpy array of features
        """
        ioutil = IOUtil('../input/laplacian_graph')
        if not stored:
            text_df = feature_df[self.text_col_name]
            corpus = text_df.values.tolist()
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_vectorizer.fit(corpus)
            constants.log_info("Prioritization of terms using TF-IDF")

            graph_list = [self.create_graph(terms, tfidf_vectorizer) for terms in corpus]
            constants.log_info("Completed graph_list")
            ioutil.to_pickle(graph_list)
            del corpus, text_df, graph_list
            return None
        else:
            graph_list = ioutil.from_pickle()
            laplacian_np_iter = ((doc_terms, degrees, edges) for doc_terms, degrees, edges in graph_list)
            constants.log_info("Completed Laplacian iter")
            laplacian = None
            for idx, laplacian_np in enumerate(laplacian_np_iter):
                laplacian = laplacian if(laplacian_np is None) else np.concatenate(laplacian, laplacian_np, axis = 0)
                constants.log_info(f'IDX: {idx}')

            return laplacian, graph_list

    def name(self) -> str:
        return FeatureEncoder.laplacian_encoder_label


    def create_graph(self, doc_terms_str: str, tfidf_vectorizer: TfidfVectorizer) -> (list, dict, dict):
        """
            Generate a graph Laplacian
            1 Select the most relevant terms using tf-idf values as ranking function
            2 Create an index for the most relevant terms (feature names)
            3 Extract the degrees and edges values for this Laplacian

            :param doc_terms_str: Terms extracted from the text defined by self.text_col_name
            :param tfidf_vectorizer: TF-IDF Vectorizer model
            :param feature_name_indices: Indices of the feature names
            :return: Tuple (doc_terms, Graph degrees, Graph edges)
        """
        from models.encoder.tfidfencoder import TfIdfEncoder

        # Step 1. Select the most relevant terms using tf-idf values as ranking function
        truncated, top_tfidf_weights = TfIdfEncoder.extract(tfidf_vectorizer, doc_terms_str, self.max_num_terms)

        feature_names = tfidf_vectorizer.get_feature_names()
        relevant_feature_names = [feature_names[idx] for idx, _ in top_tfidf_weights]

        terms_degrees = {}
        terms_edges = {}
        prev_term = None

        # Step 2. Define the indices of the relevant feature names in the document
        doc_terms = doc_terms_str.split(' ')

        # Step 3. Extract the degrees and edges values for this Laplacian
        doc_terms = (doc_term for doc_term in doc_terms if not truncated or doc_term in relevant_feature_names)

        for doc_term in doc_terms:
            if not truncated or doc_term in relevant_feature_names:
                LaplacianEncoder.__extract_degrees(terms_degrees, doc_term)
                LaplacianEncoder.__extract_edges(terms_edges, prev_term, doc_term)
                prev_term = doc_term
        del relevant_feature_names
        return doc_terms, terms_degrees, terms_edges


    def create_laplacian(self, doc_terms: list, terms_degrees: dict, terms_edges: dict) -> np.array:
        """
            Generate a Laplacian tensor with Laplacian value as first feature, normalized index for
            column doc_terms as second feature and normalized index for row as third feature.
            1 Build the Laplacian tensor
            2 Normalize the Laplacian tensor first feature

            :param doc_terms_str: Terms extracted from the text defined by self.text_col_name
            :param tfidf_vectorizer: TF-IDF Vectorizer model
            :param feature_name_indices: Indices of the feature names
            :return: 3D Numpy array
        """
        # Step 1. Build the Laplacian tensor
        laplacian_tensor = np.zeros((self.max_num_terms, self.max_num_terms, 3), dtype=np.float32)

        [self.__create_laplacian_tensor(doc_term,
                                        idx,
                                        terms_degrees,
                                        terms_edges,
                                        laplacian_tensor)
         for idx, doc_term in enumerate(doc_terms) if idx < self.max_num_terms]

        # Step 2. Normalize the Laplacian tensor first feature
        min_value = np.min(laplacian_tensor[:,0:,0])
        delta = np.max(laplacian_tensor[:,0:,0]) - min_value
        laplacian_tensor[:,0:,0] = (laplacian_tensor[:,0:,0] - min_value) / delta
        del doc_terms,
        return laplacian_tensor


    def __create_laplacian_tensor(self,
                                  term: str,
                                  row: int,
                                  terms_degrees: dict,
                                  terms_edges: dict,
                                  laplacian_tensor: np.array):
        """
            Create a Laplacian tensor value for a given doc_terms_str
            :param term: doc_terms_str in the document
            :param terms_indices: Indices of doc_terms_str in the document
            :param terms_degrees: Current degrees dictionary
            :param terms_edges: Current dictionary for the edges of the graph
            :param laplacian_tensor: Laplacian tensor to update
        """
        # Update the diagonal values for the Laplacian matrix
        if laplacian_tensor[row][row][0] == 0. and terms_degrees.get(term) is not None:
            laplacian_tensor[row][row][0] += terms_degrees.get(term)

        value = self.term_index_weights.get(term)
        if value is not None:
            laplacian_tensor[row][row][1] = value
            laplacian_tensor[row][row][2] = value
        # Update the Adjacency values for the Laplacian
        row_terms = terms_edges.get(term)
        if row_terms is not None:
            for col, row_term in enumerate(row_terms):
                if laplacian_tensor[row][col][0] == 0.0:
                    laplacian_tensor[row][col][0] -= 1.0
                laplacian_tensor[row][col][1] = value
                laplacian_tensor[row][col][2] = self.term_index_weights.get(row_term)
        else:
            constants.log_error(f'LaplacianEncoder no edges found for {term}')


    @staticmethod
    def __extract_degrees(terms_degrees: dict, term: str):
        """
            Compute the degree (diagonal value of the Laplacian) for each noed in the graph
            :param terms_degrees: Dictionary of {term, degree}
            :param term: Term used as vertex of the document graph
        """
        degree = terms_degrees.get(term)
        updated_degree = degree + 1 if degree is not None else 1
        terms_degrees.update({term: updated_degree})


    @staticmethod
    def __extract_edges(terms_edges: dict, prev_term: str, term: str):
        """
            Extract & collect edges
            :param terms_edges: Current dictionary of edges
            :param prev_term: Previous term in the sequence
            :param term: Term to be processed
        """
        if prev_term is not None:
            vertices = terms_edges.get(prev_term)
            if vertices is not None and term not in vertices:
                vertices.append(term)
                terms_edges.update({prev_term: vertices})
            else:
                terms_edges.update({prev_term: [term]})

    @staticmethod
    def __display(laplacian_tensor: np.array):
        import constants
        [constants.log_info(print(f'm[{i}][{j}][{k}] = {z}'))
         for i, x in enumerate(laplacian_tensor)
         for j, y in enumerate(x)
         for k, z in enumerate(y)]



def encode_laplacian()-> np.array:
    from models.encoder.vocabulary import Vocabulary

    df = __load_text_terms(256)
    max_num_items = 64
    term_index_weights = Vocabulary.get_index_weights()
    laplacian = LaplacianEncoder('termsText', term_index_weights, max_num_items)
    return laplacian.encode(df, False)


def __load_text_terms(num_samples: int) -> pd.DataFrame:
    from datasets.s3datasetloader import S3DatasetLoader

    s3_bucket = constants.s3_config['bucket_name']
    s3_folder = "dl/termsAndContext/utrad-all"
    col_names = ['termsText', 'age', 'gender']
    s3_dataset_loader = S3DatasetLoader(s3_bucket, s3_folder, col_names, False, '.json', num_samples)
    return s3_dataset_loader.df

if __name__ == '__main__':
    from util.profiler import Profiler

    profiler = Profiler('encode_laplacian()')
    profiler.run(20, '../output/stats-results')