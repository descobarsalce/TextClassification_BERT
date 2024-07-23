# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:00:55 2023
Edited on Mon Jul 22 2024
@author: DEscobar-Salce
"""
import tensorflow as tf
import numpy as np 
import random
from transformers import MarianMTModel, MarianTokenizer
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
import warnings
warnings.filterwarnings("ignore", message=".*max_length.*", category=UserWarning, module='transformers')

class TextDataGenerator(tf.keras.utils.Sequence):
    """
    TextDataGenerator class for generating text data with optional augmentation.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing the data.
        text_column (str): Name of the column containing the text data.
        label_column (str): Name of the column containing the labels.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the text data.
        batch_size (int): Size of the batches to generate.
        shuffle (bool): Whether to shuffle the data.
        augment_data (bool): Whether to apply data augmentation.
        always_tokenize (bool): Whether to always tokenize the text data.
    """
    
    def __init__(self, dataframe, text_column, label_column, tokenizer, batch_size, shuffle=True, augment_data=False, always_tokenize=False):
        """
        Initialize the TextDataGenerator object.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the data.
            text_column (str): Name of the column containing the text data.
            label_column (str): Name of the column containing the labels.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the text data.
            batch_size (int): Size of the batches to generate.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            augment_data (bool, optional): Whether to apply data augmentation. Defaults to False.
            always_tokenize (bool, optional): Whether to always tokenize the text data. Defaults to False.
        """
        self.dataframe = dataframe
        self.text_column = text_column
        self.label_column = label_column
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indices)
        source_lang = "en"
        target_lang = "fr"
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        self.tokenizer_translate = MarianTokenizer.from_pretrained(model_name)  # this is here to load it just once
        self.model = MarianMTModel.from_pretrained(model_name)
        model_name_back = f'Helsinki-NLP/opus-mt-{target_lang}-{source_lang}'
        self.tokenizer_back = MarianTokenizer.from_pretrained(model_name_back)
        self.model_back = MarianMTModel.from_pretrained(model_name_back)
        self.augment_data = augment_data
        self.always_tokenize = always_tokenize

    def __len__(self):
        """
        Return the number of batches per epoch.

        Returns:
            int: Number of batches per epoch.
        """
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.

        Args:
            index (int): Index of the batch.

        Returns:
            tuple: A tuple containing the input data and the labels.
        """
        if self.shuffle:
            indices = np.random.choice(len(self.dataframe), size=self.batch_size)
        else:
            indices = np.arange(index * self.batch_size, min((index + 1) * self.batch_size, len(self.dataframe)))
        batch = self.dataframe.iloc[indices]
        
        if self.augment_data:
            augmented_texts = [self.augment_text(text) for text in batch[self.text_column]]
        else:
            augmented_texts = [text for text in batch[self.text_column]]
        labels = batch[self.label_column].values
        if self.always_tokenize:
            input_data = self.tokenizer(augmented_texts, return_tensors='tf', padding=True, truncation=True, max_length=512)
            return (input_data['input_ids'], input_data['attention_mask']), labels
        else:
            return augmented_texts, labels 
        
    def on_epoch_end(self):
        """
        Shuffle the data at the end of each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def back_translate(self, text):
        """
        Perform back-translation on the input text.

        Args:
            text (str): Input text to be back-translated.

        Returns:
            str: Back-translated text.
        """
        encoded_text = self.tokenizer_translate.encode(text, return_t tensors="pt", max_length=512, truncation=True)
        translated = self.model.generate(encoded_text, pad_token_id=0)
        target_text = self.tokenizer_translate.decode(translated[0], skip_special_tokens=True)
        encoded_target_text = self.tokenizer_back.encode(target_text, return_tensors="pt", max_length=512, truncation=True)
        translated_back = self.model_back.generate(encoded_target_text, pad_token_id=0)
        source_text = self.tokenizer_back.decode(translated_back[0], skip_special_tokens=True)
        return source_text

    @staticmethod
    def synonym_replacement(text, num_replacements=2):
        """
        Replace words in the text with their synonyms.

        Args:
            text (str): Input text.
            num_replacements (int, optional): Number of words to replace. Defaults to 2.

        Returns:
            str: Text with synonyms replaced.
        """
        words = text.split()
        new_words = words.copy()
        for _ in range(num_replacements):
            word_to_replace = random.choice(words)
            synonyms = wordnet.synsets(word_to_replace)
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name()
                new_words = [synonym if word == word_to_replace else word for word in new_words]
        return ' '.join(new_words)

    def random_insertion(self, text, num_insertions=2):
        """
        Insert random synonyms into the text.

        Args:
            text (str): Input text.
            num_insertions (int, optional): Number of words to insert. Defaults to 2.

        Returns:
            str: Text with random insertions.
        """
        words = text.split()
        for _ in range(num_insertions):
            synonym_word = self.synonym_replacement(random.choice(words)).split()[0]
            random_idx = random.randint(0, len(words) - 1)
            words.insert(random_idx, synonym_word)
        return ' '.join(words)

    @staticmethod
    def random_deletion(text, p=0.2):
        """
        Randomly delete words from the text.

        Args:
            text (str): Input text.
            p (float, optional): Probability of deleting each word. Defaults to 0.2.

        Returns:
            str: Text with random deletions.
        """
        words = text.split()
        if len(words) == 1:
            return words
        remaining = list(filter(lambda x: random.uniform(0, 1) > p, words))
        return ' '.join(remaining if remaining else [random.choice(words)])

    def augment_text(self, text, methods=['back_translate', 'synonym_replacement']):
        """
        Apply text augmentation methods to the input text.

        Args:
            text (str): Input text.
            methods (list, optional): List of augmentation methods to apply. Defaults to ['back_translate', 'synonym_replacement'].

        Returns:
            str: Augmented text.
        """
        if 'back_translate' in methods:
            text = self.back_translate(text)
        if 'synonym_replacement' in methods:
            text = self.synonym_replacement(text)
        if 'random_insertion' in methods:
            text = self.random_insertion(text)
        if 'random_deletion' in methods:
            text = self.random_deletion(text)
        return text
    
    def augment_text_batch(self, texts):
        """
        Apply text augmentation methods to a batch of texts.

        Args:
            texts (list): List of input texts.

        Returns:
            list: List of augmented texts.
        """
        augmented_texts = [self.augment_text(text) for text in texts]
        return augmented_texts
