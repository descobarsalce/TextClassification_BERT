# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 2023
Edited on Mon Jul 22 2024
@author: DEscobar-Salce
"""

from transformers import BartForConditionalGeneration, BartTokenizer

class SummarizerBART:
    """
    SummarizerBART class for text summarization using the BART model.
    """

    def __init__(self, model="facebook/bart-large-cnn"):
        """
        Initialize the SummarizerBART object.

        Args:
            model (str, optional): Pretrained BART model to use for summarization. Defaults to 'facebook/bart-large-cnn'.
        """
        self.model_name = model
        self.bart_model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.bart_tokenizer = BartTokenizer.from_pretrained(self.model_name)
        
    def summarize_text(self, text, min_length=400, max_length=512, num_beams=4):
        """
        Summarize the input text using the BART model.

        Args:
            text (str): The input text to summarize.
            min_length (int, optional): Minimum length of the summary. Defaults to 400.
            max_length (int, optional): Maximum length of the summary. Defaults to 512.
            num_beams (int, optional): Number of beams for beam search. Defaults to 4.

        Returns:
            str: The summarized text.
        """
        inputs = self.bart_tokenizer([text], max_length=max_length, return_tensors='pt', truncation=True)
        summary_ids = self.bart_model.generate(inputs.input_ids, num_beams=num_beams, min_length=min_length,
                                               max_length=max_length, early_stopping=True)
        return self.bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
