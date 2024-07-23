# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 06:57:55 2023
Edited on Mon Jul 22 2024
@author: DEscobar-Salce
"""

import tensorflow as tf
from src.BART_Summarizer import SummarizerBART
from transformers import BertTokenizer, DistilBertTokenizer
from tqdm import tqdm
import regex as re

class StringPreprocessing:
    """
    StringPreprocessing class for handling text preprocessing, including tokenization and summarization.
    """

    def __init__(self, model='distilbert-base-uncased'):
        """
        Initialize the StringPreprocessing object.

        Args:
            model (str, optional): Pretrained BERT model to use for tokenization. Defaults to 'distilbert-base-uncased'.
        """
        if model == 'distilbert-base-uncased':
            self.tokenizer = DistilBertTokenizer.from_pretrained(model, do_lower_case=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=True)
        self.summarizer = SummarizerBART(model="facebook/bart-large-cnn")

    def verify_length_and_summarize(self, df, col_name, new_comment_name='summarized_comment'):
        """
        Check the length of text entries and summarize those that exceed the BERT token limit.

        Args:
            df (DataFrame): DataFrame containing the text data.
            col_name (str): Name of the column containing the text data.
            new_comment_name (str, optional): Name of the new column to store summarized text. Defaults to 'summarized_comment'.

        Returns:
            DataFrame: DataFrame with summarized text for long entries.
        """
        df['split_length'] = df[col_name].str.split().str.len()
        short_string_data = df[df['split_length'] < 500]  # Word count rather than tokens but should work
        long_string_data = df[df['split_length'] >= 500]
        tqdm.pandas()
        long_string_data[new_comment_name] = long_string_data[col_name].progress_apply(
            lambda x: self.summarizer.summarize_text(x, min_length=400, max_length=512, num_beams=4))
        return short_string_data.append(long_string_data)
        
    def tokenize(self, df, col_name, summarize=False):
        """
        Tokenize the text data using the BERT tokenizer.

        Args:
            df (DataFrame): DataFrame containing the text data.
            col_name (str): Name of the column containing the text data.
            summarize (bool, optional): Whether to summarize long text entries. Defaults to False.

        Returns:
            tuple: Tuple containing the tokenized input IDs and attention masks.
        """
        if summarize:
            print("Generating summaries for long comments:")
            new_comment_name = 'summarized_comment'
            df = self.verify_length_and_summarize(df, col_name, new_comment_name=new_comment_name)
            col_name = new_comment_name

        tokens = self.tokenizer.batch_encode_plus(df[col_name].tolist(),
                                                  add_special_tokens=True,
                                                  max_length=512,
                                                  padding='max_length',
                                                  truncation=True,
                                                  return_attention_mask=True,
                                                  return_token_type_ids=False,
                                                  return_tensors='tf')
        input_ids = tf.constant(tokens['input_ids'])
        attention_masks = tf.constant(tokens['attention_mask'])

        return input_ids, attention_masks

    @staticmethod
    def clean_string(string):
        """
        Clean a string by applying various text transformations.

        Args:
            string (str): The input string to clean.

        Returns:
            str: The cleaned string.
        """
        string = str(string)
        string = string.encode("ascii", errors="ignore").decode()
        string = string.lower()
        chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
        string = re.sub(rx, '', string)
        string = string.replace('&', 'and')
        string = string.replace(',', ' ')
        string = string.replace('-', ' ')
        string = re.sub(' +', ' ', string).strip()
        string = ' ' + string + ' '
        string = re.sub(r'[,-./]|\sBD', r'', string)
        string = re.sub('\s+', ' ', string)
        return string
