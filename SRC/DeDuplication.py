# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:11:01 2023
Edited on Mon Jul 22 2024
@author: DEscobar-Salce
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import sparse_dot_topn.sparse_dot_topn as ct
from scipy.sparse import csr_matrix
import time
import pandas as pd
import regex as re
import numpy as np
import networkx as nx

class RecordLinkage:
    """
    A class to perform record linkage and de-duplication on a dataset using string comparison techniques.

    Attributes:
        df_messy_original (pandas.DataFrame): Original DataFrame with messy names.
        df_messy (pandas.DataFrame): DataFrame with unique messy names.
        messy_names_var (str): Name of the variable/column containing the messy names.
    """

    def __init__(self, df_messy, messy_names_var):
        """
        Initialize the RecordLinkage class.

        Parameters:
            df_messy (pandas.DataFrame): DataFrame with inconsistent names.
            messy_names_var (str): Name of the variable/column containing the messy names.
        """
        self.df_messy_original = df_messy[[messy_names_var]]
        self.df_messy = self.df_messy_original.drop_duplicates()
        self.messy_names_var = messy_names_var

    def clean_all_strings(self):
        """
        Clean all strings in df_messy.
        """
        self.df_messy = self.df_messy.apply(RecordLinkage.clean_string)

    def deduplication(self, ntop=5, lower_bound=0.8, max_matches=1000, n_grams=[3, 4], full_words=True):
        """
        Perform de-duplication on a DataFrame using TF-IDF similarity.

        Parameters:
            ntop (int): Number of top matches to consider for each name (default: 5).
            lower_bound (float): Lower bound similarity threshold for matches (default: 0.8).
            max_matches (int): Number of top matches to include in the output DataFrame (default: 1000).

        Returns:
            pandas.DataFrame: DataFrame containing the top matching pairs with similarity scores.
        """
        names_df = pd.DataFrame(self.df_messy[self.messy_names_var].unique())[0]
        names_df.drop_duplicates(inplace=True)

        vector_transformer = TfidfVectorizer(min_df=1,
                                             analyzer=lambda x: RecordLinkage.ngrams(
                                                 x, n_sizes=n_grams, full_words=full_words),
                                             lowercase=True)
        t1 = time.time()
        tf_idf_matrix = vector_transformer.fit_transform(names_df)
        matches = RecordLinkage.cosine_similarity(
            tf_idf_matrix, tf_idf_matrix.transpose(), ntop, lower_bound)
        matches_df = RecordLinkage.get_matches_df_duplicates(matches, names_df, top=max_matches)
        matches_df.drop_duplicates()
        t = time.time() - t1
        print("SELFTIMED:", t)
        return matches_df

    @staticmethod
    def getNearestN(vector_transformer, nbrs, query):
        """
        Get the nearest neighbors of a query using the trained vectorizer and NearestNeighbors model.

        Parameters:
            nbrs (sklearn.neighbors.NearestNeighbors): Trained NearestNeighbors model.
            query (list or str): Query or list of queries to find nearest neighbors.

        Returns:
            numpy.ndarray: Distances to the nearest neighbors.
            numpy.ndarray: Indices of the nearest neighbors.
        """
        queryTFIDF_ = vector_transformer.transform(query)
        distances, indices = nbrs.kneighbors(queryTFIDF_)
        return distances, indices

    @staticmethod
    def ngrams(string, n_sizes=[3, 4], full_words=True):
        """
        Generate n-grams from a string of all the sizes in n_sizes.

        Parameters:
            string (str): The input string to generate n-grams from.
            n_sizes (list of int): List of desired n-grams lengths to include.
            full_words (bool): Determines whether to include full words in n-grams list.

        Returns:
            list: A list of n-grams.
        """
        if string:  # handling missing values
            all_ngrams = []
            for size in n_sizes:
                ngrams = zip(*[string[i:] for i in range(size)])
                all_ngrams = all_ngrams + [''.join(ngram) for ngram in ngrams]
            if full_words:
                all_ngrams = all_ngrams + string.split(" ")
            return all_ngrams
        return ['']  # This is only for empty strings

    @staticmethod
    def clean_string(string):
        """
        Clean a string by applying various text transformations.

        Parameters:
            string (str): The input string to clean.

        Returns:
            str: The cleaned string.
        """
        string = str(string)
        string = string.encode("ascii", errors="ignore").decode()  # Remove non-ASCII characters
        string = string.lower()  # Convert to lowercase
        chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
        string = re.sub(rx, '', string)  # Remove specified characters
        string = string.replace('&', 'and')  # Replace '&' with 'and'
        string = string.replace(',', ' ')  # Replace commas with spaces
        string = string.replace('-', ' ')  # Replace dashes with spaces
        string = re.sub(' +', ' ', string).strip()  # Replace multiple spaces with a single space
        string = ' ' + string + ' '  # Pad names for n-grams
        string = re.sub(r'[,-./]|\sBD', r'', string)
        string = re.sub('\s+', ' ', string)
        return string

    @staticmethod
    def clean_up_words(self, test, new_names_var_clean_up, departments, remove_departments=True):
        """
        Clean up words in a DataFrame column by removing specific words and connectors.

        Parameters:
            test (pandas.DataFrame): The DataFrame to clean.
            new_names_var_clean_up (str): Name of the column to clean.
            departments (list): List of department names.
            remove_departments (bool): Whether to remove department names (default: True).

        Returns:
            pandas.DataFrame: The cleaned DataFrame.
        """
        words_to_delete = ['regents', 'regent', 'foundation', 'endowment', 'fund', 'fellows',
                           'president', 'board of trustees', 'board', 'trustees', 'trust', 'scholarship', 'fellowship']
        connectors = [' of ', ' in ', ' of ', ' and ', ' for ']
        university_words = set(
            ['university', 'community college', 'college', 'institute', 'academy', 'school'])
        school_words_to_delete = [
            'graduate school', 'college', 'school', 'graduate', 'department', 'center']
        for connector in connectors:
            test[new_names_var_clean_up] = test[new_names_var_clean_up].str.replace(
                connector, " ")

        for word in words_to_delete:
            test[new_names_var_clean_up] = test[new_names_var_clean_up].apply(
                lambda x: x.replace(" " + word + " ", " "))

        if remove_departments:
            for word in school_words_to_delete:
                for department in departments:
                    test[new_names_var_clean_up] = test[new_names_var_clean_up].apply(lambda x: self.replace_if_not_contain(
                        x, university_words, " " + word + " " + department + " ", " 0987654321 "))
                    test[new_names_var_clean_up] = test[new_names_var_clean_up].apply(lambda x: self.replace_if_not_contain(
                        x, university_words, " " + department + " " + word + " ", " 0987654321 "))

        return test

    @staticmethod
    def cosine_similarity(A, B, ntop, lower_bound=0):
        """
        Calculate the top cosine similarity matches between two sparse matrices A and B.

        Parameters:
            A (scipy.sparse.csr_matrix): Sparse matrix A.
            B (scipy.sparse.csr_matrix): Sparse matrix B.
            ntop (int): Number of top matches to return.
            lower_bound (float): Lower bound threshold for similarity.

        Returns:
            scipy.sparse.csr_matrix: Sparse matrix containing the top similarity matches.
        """
        A = A.tocsr()
        B = B.tocsr()
        M, _ = A.shape
        _, N = B.shape
        idx_dtype = np.int32
        nnz_max = M * ntop

        indptr = np.zeros(M + 1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)

        ct.sparse_dot_topn(
            M, N,
            np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data
        )

        return csr_matrix((data, indices, indptr), shape=(M, N))

    @staticmethod
    def get_matches_df_duplicates(sparse_matrix, name_vector, top=100):
        """
        Create a DataFrame of matches from a sparse matrix of similarities.

        Parameters:
            sparse_matrix (scipy.sparse.csr_matrix): Sparse matrix of similarities.
            name_vector (numpy.ndarray): Array of names corresponding to the sparse matrix rows/columns.
            top (int): Number of top matches to include in the DataFrame.

        Returns:
            pandas.DataFrame: DataFrame containing the matches with left and right names and similarity scores.
        """
        non_zeros = sparse_matrix.nonzero()
        sparserows = non_zeros[0]
        sparsecols = non_zeros[1]

        if top:
            nr_matches = min(top, sparsecols.size)
        else:
            nr_matches = sparsecols.size

        left_side = np.empty([nr_matches], dtype=object)
        right_side = np.empty([nr_matches], dtype=object)
        similarity = np.zeros(nr_matches)

        for index in range(0, nr_matches):
            left_side[index] = name_vector[sparserows[index]]
            right_side[index] = name_vector[sparsecols[index]]
            similarity[index] = sparse_matrix.data[index]

        new_df = pd.DataFrame({'left_side': left_side,
                               'right_side': right_side,
                               'similarity': similarity})
        
        return new_df[new_df.left_side != new_df.right_side]
    
    def generate_group_ids(self, deduplicated):
        """
        Generate group IDs for deduplicated entries.

        Parameters:
            deduplicated (pandas.DataFrame): DataFrame containing deduplicated entries.

        Returns:
            pandas.DataFrame: DataFrame with original entries and their corresponding group IDs.
        """
        deduplicated = deduplicated.dropna(subset=['left_side', 'right_side'])
        G = nx.from_pandas_edgelist(deduplicated, 'left_side', 'right_side')
        connected_components = list(nx.connected_components(G))
        group_dict = {}
        for group_id, component in enumerate(connected_components, start=1):
            for entry in component:
                group_dict[entry] = group_id
                
        deduplicated['group_id'] = deduplicated['left_side'].map(group_dict).combine_first(deduplicated['right_side'].map(group_dict))
        deduplicated_codes = deduplicated[['left_side', 'group_id']].drop_duplicates(inplace=False)
        
        dedup_merges = pd.merge(self.df_messy, deduplicated_codes, 
                                left_on=self.messy_names_var, 
                                right_on='left_side', 
                                how='outer')
        max_group_id = dedup_merges['group_id'].max()
        missing_groups = dedup_merges['group_id'].isnull().sum()
        if missing_groups > 0:
            new_group_ids = range(int(max_group_id) + 1, int(max_group_id) + missing_groups + 1)
            dedup_merges.loc[dedup_merges['group_id'].isnull(), 'group_id'] = new_group_ids
        
        return dedup_merges[[self.messy_names_var, 'group_id']]
    
    def similarity_at_K(self, threshold, deduplicated, draft_entries, known_campaigns):
        """
        Calculate similarity at a given threshold and generate group IDs.

        Parameters:
            threshold (float): Similarity threshold.
            deduplicated (pandas.DataFrame): DataFrame containing deduplicated entries.
            draft_entries (pandas.DataFrame): DataFrame containing draft entries.
            known_campaigns (pandas.DataFrame): DataFrame containing known campaigns.

        Returns:
            pandas.DataFrame: DataFrame with similarity information and group IDs.
        """
        deduplicated = deduplicated[deduplicated.similarity > threshold]
        new_group_ids = self.generate_group_ids(deduplicated)
        new_group_ids['repetitions_count'] = new_group_ids.groupby('group_id')['group_id'].transform('count')
        print(new_group_ids['repetitions_count'].value_counts(dropna=False).sort_index())
        
        draft_entries_ids = pd.merge(draft_entries, new_group_ids, 
                                     how='left', 
                                     on='comment')
        draft_entries_ids.repetitions_count.value_counts(dropna=False)
        
        known_campaigns_ids = pd.merge(known_campaigns, new_group_ids, 
                                       how='left', 
                                       on='comment')
        known_campaigns_ids = known_campaigns_ids[['entry_code_parent', 'group_id', 'repetitions_count']]
        known_campaigns_ids.rename(columns={'repetitions_count': 'repetitions_count_knowncamp'}, inplace=True)
        known_campaigns_ids = known_campaigns_ids[['group_id', 'entry_code_parent', 'repetitions_count_knowncamp']]
        
        draft_entries_ids = pd.merge(draft_entries_ids, known_campaigns_ids, 
                                     how='left', 
                                     on='group_id',
                                     indicator='known_campaign')
        
        draft_entries_ids['mass_campaign'] = ((draft_entries_ids.known_campaign == 'both') | (draft_entries_ids.repetitions_count > 3))
        
        draft_entries_ids = draft_entries_ids[['double_key', 'entry_code', 'file_name', 'comment', 'group_id', 'repetitions_count', 'entry_code_parent', 'mass_campaign']]
            
        return draft_entries_ids
