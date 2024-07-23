#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 23:19:23 2023
Edited on Mon Jul 22 2024
@author: descobarsalce
"""

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import TFBertForSequenceClassification, TFDistilBertForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle
from tqdm import tqdm
from src.ProgressBar import ProgressBar
from src.BERT_Preprocessing import StringPreprocessing
from src.TextDataGenerator import TextDataGenerator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool

class ClassifierBERT:
    """
    ClassifierBERT class for training and evaluating a BERT-based text classification model.
    """

    def __init__(self, labeled=None, text_varname=None, labels_varname=None, model='distilbert-base-uncased', 
                 trained_model=False, description_mapping=None):
        """
        Initialize the ClassifierBERT object.

        Args:
            labeled (pd.DataFrame): Labeled data for training the model.
            text_varname (str): Name of the column containing the text data.
            labels_varname (str): Name of the column containing the labels.
            model (str, optional): Pretrained BERT model to use. Defaults to 'distilbert-base-uncased'.
            trained_model (bool, optional): Whether a trained model is being loaded. Defaults to False.
            description_mapping (dict, optional): Mapping of label descriptions. Defaults to None.
        """
        if not trained_model:
            try:
                assert isinstance(labeled, pd.DataFrame), "labeled must be a DataFrame."
                assert isinstance(text_varname, str), "text_varname must be a string."
                assert isinstance(labels_varname, str), "labels_varname must be a string."
            except AssertionError as e:
                print(f"Error during initialization: {e}")
                return
            if model == 'bert-base-uncased':
                self.bert_model = TFBertForSequenceClassification.from_pretrained(model)
            elif model == 'distilbert-base-uncased':
                self.bert_model = TFDistilBertForSequenceClassification.from_pretrained(model)
            else:
                print("Error, model not found. Please choose 'bert-base-uncased' or 'distilbert-base-uncased'.")
        else:
            self.model = None

        self.TextPreprocessor = StringPreprocessing()
        self.description_mapping = description_mapping
        self.data_labeled = labeled.copy()
        self.text_varname = text_varname
        self.labels_varname = labels_varname
        self.history = []
        self.mlb = MultiLabelBinarizer()
        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.model = None
        self.model_name = model 

    def prepare_sample(self, train_size, test_size, val_size, seed, data_augmentation=False, N=0):
        """
        Prepare the sample for training, validation, and testing.

        Args:
            train_size (float): Proportion of data for training.
            test_size (float): Proportion of data for testing.
            val_size (float): Proportion of data for validation.
            seed (int): Random seed for reproducibility.
            data_augmentation (bool, optional): Whether to apply data augmentation. Defaults to False.
            N (int, optional): Number of augmented samples to generate. Defaults to 0.
        """
        if not isinstance(self.train_df, pd.DataFrame):
            self.train_tst_val_split(train_size=train_size, test_size=test_size, val_size=val_size, random_state=seed)
    
    def data_augmentation(self, N, batch_size=16, shuffle=True):
        """
        Apply data augmentation to the training data.

        Args:
            N (int): Number of augmented samples to generate.
            batch_size (int, optional): Batch size for data augmentation. Defaults to 16.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

        Returns:
            pd.DataFrame: Augmented data.
        """
        if N > 0:
            print('Starting data augmentation:')
            self.train_data_generator = TextDataGenerator(self.train_df.copy(), 
                                                          text_column=self.text_varname, 
                                                          label_column=self.labels_varname, 
                                                          tokenizer=None,
                                                          batch_size=batch_size, 
                                                          shuffle=shuffle,
                                                          augment_data=True,
                                                          always_tokenize=False)
            new_data = ClassifierBERT.generate_augmented_data_with_progress_bar(
                train_data_generator=self.train_data_generator, N=N, batch_size=batch_size, shuffle=shuffle)
            return new_data
    
    def compile_model_multilabel(self, threshold=0.5, initial_learning_rate=3e-3, epsilon=1e-08, 
                                 clipnorm=1.0, dropout_rate=0.2, l1_regularization_alpha=0.01, sequence_length=512):
        """
        Compile the BERT model with the specified parameters for multi-label classification.

        Args:
            threshold (float, optional): Threshold for classification. Defaults to 0.5.
            initial_learning_rate (float, optional): Initial learning rate. Defaults to 3e-3.
            epsilon (float, optional): Epsilon value for the optimizer. Defaults to 1e-08.
            clipnorm (float, optional): Gradient clipping norm. Defaults to 1.0.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
            l1_regularization_alpha (float, optional): L1 regularization alpha. Defaults to 0.01.
            sequence_length (int, optional): Sequence length for input. Defaults to 512.
        """
        input_ids = tf.keras.layers.Input(shape=(sequence_length,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(sequence_length,), dtype=tf.int32, name='attention_mask')

        if isinstance(self.bert_model, TFBertForSequenceClassification):
            short_model_name = 'bert'
        elif isinstance(self.bert_model, TFDistilBertForSequenceClassification):
            short_model_name = 'distilbert'
        else:
            print("The model is neither BERT nor DistilBERT.")
            raise ValueError("The model is neither BERT nor DistilBERT.")
        
        bert_layer_output = self.bert_model.get_layer(short_model_name)([input_ids, attention_mask])
        
        if isinstance(self.bert_model, TFBertForSequenceClassification):
            cls_output = bert_layer_output.pooler_output
        elif isinstance(self.bert_model, TFDistilBertForSequenceClassification):
            cls_output = tf.keras.layers.Lambda(lambda x: x[:, 0, :], name='model_layer_output')(bert_layer_output.last_hidden_state)
        
        bert_preclassifier = self.bert_model.get_layer('pre_classifier')(cls_output)
        bert_classifier = self.bert_model.get_layer('classifier')(bert_preclassifier)
        bert_classifier_w_dropout = tf.keras.layers.Dropout(0.5, name='my_classifier_dropout')(bert_classifier)
        num_labels = len(self.data_labeled[self.labels_varname].iloc[0])
        regularization_layer = tf.keras.layers.Dense(num_labels * 8, kernel_regularizer=tf.keras.regularizers.l1(l1_regularization_alpha), name="regularization_layer")(bert_classifier_w_dropout)

        output_layer = tf.keras.layers.Dense(num_labels, activation='sigmoid', name='final_classifier')(regularization_layer)

        self.model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output_layer, name='classification_model')
        
        decay_at_epochs = [5, 10, 15, 20, 25, 30, 35]
        decay_factor = 0.8
        values = [initial_learning_rate]
        for _ in range(len(decay_at_epochs)):
            values.append(values[-1] * decay_factor)
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
              boundaries=decay_at_epochs, values=values)

        optimizer = tf.keras.optimizers.Adam(
              learning_rate=lr_schedule, epsilon=epsilon, clipnorm=clipnorm)

        def weighted_binary_crossentropy(y_true, y_pred):
            false_negative_weight = 1.0
            false_positive_weight = 1.0
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
            bce = -(y_true * tf.math.log(y_pred) * false_negative_weight + (1 - y_true) * tf.math.log(1 - y_pred) * false_positive_weight)
            return tf.reduce_mean(bce)

        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = []
        average = 'micro'
        metrics.append(tfa.metrics.F1Score(
            num_classes=num_labels, average=average, threshold=threshold))
        metrics.append(tf.keras.metrics.Precision(name='precision', thresholds=threshold))
        metrics.append(tf.keras.metrics.Recall(name='recall', thresholds=threshold))
        metrics.append(tf.keras.metrics.TruePositives(name='true_positives'))
        metrics.append(tf.keras.metrics.FalsePositives(name='false_positives'))
        metrics.append(tf.keras.metrics.TrueNegatives(name='true_negatives'))
        metrics.append(tf.keras.metrics.FalseNegatives(name='false_negatives'))
        metrics.append(tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'))
        metrics.append(tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=threshold))

        self.model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy, metrics=metrics)
    
    def train_model(self, epochs=2, batch_size=32, train_size=0.8, test_size=0.1, val_size=0.1,
                    show_confusion_matrix=True, show_roc_auc=True, seed=123, metric='accuracy',
                    classification_type='multilabel', threshold=0.5, initial_learning_rate=2e-5, 
                    epsilon=1e-08, data_augmentation=False, N=0, early_stopping_patience=100):
        """
        Train the BERT model on the labeled data.

        Args:
            epochs (int, optional): Number of training epochs. Defaults to 2.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            train_size (float, optional): Proportion of data for training. Defaults to 0.8.
            test_size (float, optional): Proportion of data for testing. Defaults to 0.1.
            val_size (float, optional): Proportion of data for validation. Defaults to 0.1.
            show_confusion_matrix (bool, optional): Whether to show the confusion matrix. Defaults to True.
            show_roc_auc (bool, optional): Whether to show the ROC curve and AUC score. Defaults to True.
            seed (int, optional): Random seed for reproducibility. Defaults to 123.
            metric (str, optional): Evaluation metric to use. Defaults to 'accuracy'.
            classification_type (str, optional): Type of classification ('multilabel' or 'multiclass'). Defaults to 'multilabel'.
            threshold (float, optional): Threshold for classification. Defaults to 0.5.
            initial_learning_rate (float, optional): Initial learning rate. Defaults to 2e-5.
            epsilon (float, optional): Epsilon value for the optimizer. Defaults to 1e-08.
            data_augmentation (bool, optional): Whether to apply data augmentation. Defaults to False.
            N (int, optional): Number of augmented samples to generate. Defaults to 0.
            early_stopping_patience (int, optional): Patience for early stopping. Defaults to 100.
        """
        if not isinstance(self.train_df, pd.DataFrame):
            self.prepare_sample(train_size, test_size, val_size, seed)

        datasets = [self.train_df, self.val_df, self.test_df]
        results = []
        print("Generating tokens.")
        for dataset in datasets:
            input_ids, attention_masks = self.TextPreprocessor.tokenize(dataset, self.text_varname)
            labels = tf.constant(dataset[self.labels_varname].tolist())
            results.append((input_ids, attention_masks, labels))
        input_ids_tr, attention_masks_tr, labels_tr = results[0]
        input_ids_val, attention_masks_val, labels_val = results[1]
        input_ids_test, attention_masks_test, labels_test = results[2]

        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=early_stopping_patience, 
            verbose=1, 
            restore_best_weights=True
        )
        
        print("Model training started:")
        if tf.test.gpu_device_name():
            with tf.device('/GPU:0'):
                new_history = self.model.fit(x=(input_ids_tr, attention_masks_tr),
                                             y=labels_tr,
                                             epochs=epochs,
                                             batch_size=batch_size,
                                             validation_data=((input_ids_val, attention_masks_val), labels_val),
                                             callbacks=[early_stopping],
                                             shuffle=True)
                self.history.append(new_history)
        else:
            new_history = self.model.fit(x=(input_ids_tr, attention_masks_tr),
                                         y=labels_tr,
                                         epochs=epochs,
                                         batch_size=batch_size,
                                         validation_data=((input_ids_val, attention_masks_val), labels_val),
                                         callbacks=[early_stopping],
                                         shuffle=True)
            self.history.append(new_history)
        self.test_model_performance(input_ids_test, attention_masks_test, labels_test, classification_type, 
                                    threshold, show_confusion_matrix, show_roc_auc)
        
    def test_model_performance(self, input_ids_test, attention_masks_test, labels_test, classification_type, 
                               threshold, show_confusion_matrix, show_roc_auc):
        """
        Test the performance of the trained model.

        Args:
            input_ids_test (tf.Tensor): Input IDs for the test set.
            attention_masks_test (tf.Tensor): Attention masks for the test set.
            labels_test (tf.Tensor): True labels for the test set.
            classification_type (str): Type of classification ('multilabel' or 'multiclass').
            threshold (float): Threshold for classification.
            show_confusion_matrix (bool): Whether to show the confusion matrix.
            show_roc_auc (bool): Whether to show the ROC curve and AUC score.
        """
        input_data_test = {
            'input_ids': input_ids_test,
            'attention_mask': attention_masks_test
        }
        predicted_prob = self.model.predict(input_data_test)

        if classification_type == 'multilabel':
            predicted_labels = (predicted_prob > threshold).astype(int)
        elif classification_type == 'multiclass': 
            predicted_labels = np.argmax(predicted_prob, axis=1)

        self.show_model_performance(predicted_prob, labels_test, predicted_labels,
                                    show_confusion_matrix=show_confusion_matrix, show_roc_auc=show_roc_auc)
        
    def show_model_performance(self, predictions, labels_test, predicted_labels, show_confusion_matrix=True,
                               show_roc_auc=True, classification_type='multilabel'):
        """
        Show the performance of the trained model.

        Args:
            predictions (tf.Tensor): Predicted probabilities from the model.
            labels_test (tf.Tensor): True labels from the testing set.
            predicted_labels (np.ndarray): Predicted labels from the model.
            show_confusion_matrix (bool, optional): Whether to show the confusion matrix. Defaults to True.
            show_roc_auc (bool, optional): Whether to show the ROC curve and AUC score. Defaults to True.
            classification_type (str, optional): Type of classification ('multilabel' or 'multiclass'). Defaults to 'multilabel'.
        """
        if classification_type == 'multilabel':        
            micro_f1 = f1_score(labels_test, predicted_labels, average='micro')
            macro_f1 = f1_score(labels_test, predicted_labels, average='macro')
            print(f"Micro F1 Score: {micro_f1}")
            print(f"Macro F1 Score: {macro_f1}")
            
            if show_roc_auc:
                num_classes = labels_test.shape[1]
                for i in range(num_classes):
                    y_true_class = labels_test[:, i]
                    y_pred_class = predicted_labels[:, i]
                    if sum(y_true_class.numpy()) > 0:
                        fpr, tpr, _ = roc_curve(y_true_class, y_pred_class)
                        auc = roc_auc_score(y_true_class, y_pred_class)
                        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc:.3f})')
            
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc='lower right')
                plt.show()
                
        elif classification_type == 'multiclass': 
            if show_confusion_matrix:
                cm = confusion_matrix(labels_test, predicted_labels)
                print(cm)
            if show_roc_auc:
                y_pred_proba = tf.nn.softmax(predictions.logits, axis=-1)[:, 1]
                fpr, tpr, _ = roc_curve(labels_test, y_pred_proba)
                auc = roc_auc_score(labels_test, y_pred_proba)
                print(auc)
                plt.plot(fpr, tpr)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve (AUC = {:.3f})')
                plt.show()

    def predict_labels(self, unlabeled_df, unlabeled_text_varname, threshold=0.5):
        """
        Predict labels for unlabeled data.

        Args:
            unlabeled_df (pd.DataFrame): Unlabeled data for prediction.
            unlabeled_text_varname (str): Name of the column containing the text data.
            threshold (float, optional): Threshold for classification. Defaults to 0.5.

        Returns:
            tuple: Tuple containing the predicted probabilities and labels.
        """
        input_ids_unlab, attention_masks_unlab = self.TextPreprocessor.tokenize(unlabeled_df[[unlabeled_text_varname]], unlabeled_text_varname)
        input_data = {
            'input_ids': input_ids_unlab,
            'attention_mask': attention_masks_unlab
        }
        predictions_unlab = self.model.predict(input_data)
        assigned_categories = (predictions_unlab > threshold).astype(int)
        return predictions_unlab, assigned_categories

    @staticmethod
    def generate_readable_labels(unlabeled_data_unique, predictions, labels, description_mapping):
        """
        Convert numeric labels to human-readable labels.

        Args:
            unlabeled_data_unique (pd.DataFrame): DataFrame with unique unlabeled data.
            predictions (np.ndarray): Predicted probabilities.
            labels (np.ndarray): Predicted labels.
            description_mapping (dict): Mapping of label descriptions.

        Returns:
            pd.DataFrame: DataFrame with human-readable labels.
        """
        all_labels = sorted([int(k) for k in description_mapping.keys()])
        human_readable_labels = []
        for row in labels:
            categories_for_row = [description_mapping[str(all_labels[col])] for col, val in enumerate(row) if val == 1]
            human_readable_labels.append(categories_for_row)
        new_data = pd.concat([pd.DataFrame({'logits_values': predictions.tolist()})['logits_values'],
                              pd.DataFrame({'numeric_labels': labels.tolist()})['numeric_labels'],
                              pd.DataFrame({'human_readable_labels': human_readable_labels})['human_readable_labels']
                              ], axis=1)
        unlabeled_data_unique.reset_index(inplace=True, drop=True)
        new_data = pd.concat([unlabeled_data_unique, new_data], axis=1)
        return new_data

    def predict_unlabeled_data(self, unlabeled_data_unique, unlabeled_text_varname):
        """
        Predict labels for unlabeled data and generate human-readable labels.

        Args:
            unlabeled_data_unique (pd.DataFrame): DataFrame with unique unlabeled data.
            unlabeled_text_varname (str): Name of the column containing the text data.

        Returns:
            pd.DataFrame: DataFrame with human-readable labels.
        """
        predictions, labels = self.predict_labels(unlabeled_data_unique, unlabeled_text_varname)
        df_human_readable_labels = ClassifierBERT.generate_readable_labels(unlabeled_data_unique, predictions, labels, self.description_mapping)
        return df_human_readable_labels

    def show_epoch_evolution(self):
        """
        Show the evolution of training and validation metrics across epochs.
        """
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def freeze_BERT_layers(self, list_layername_to_freeze, freeze=True):
        """
        Freeze or unfreeze specified BERT layers.

        Args:
            list_layername_to_freeze (list): List of layer names to freeze or unfreeze.
            freeze (bool, optional): Whether to freeze the layers. Defaults to True.
        """
        for layer in self.model.layers:      
            if "bert" in layer.name or layer.name in list_layername_to_freeze:
                layer.trainable = not freeze

    def save_model(self, path):
        """
        Save the model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        self.model.save(path)
        history_path = os.path.join(path, 'history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.history, f)

    @classmethod
    def load_model(cls, path, *args, **kwargs):
        """
        Load a model from the specified path.

        Args:
            path (str): Path to load the model from.

        Returns:
            ClassifierBERT: Loaded ClassifierBERT object.
        """
        LoadedClassifier = ClassifierBERT(trained_model=True)
        LoadedClassifier.model = load_model(path)
        history_path = os.path.join(path, 'history.pkl')
        with open(history_path, 'rb') as f:
            LoadedClassifier.history = pickle.load(f)
        return LoadedClassifier

    def train_tst_val_split(self, train_size=0.8, test_size=0.1, val_size=0.1, random_state=111):
        """
        Split the labeled data into training, testing, and validation sets.

        Args:
            train_size (float, optional): Proportion of data for training. Defaults to 0.8.
            test_size (float, optional): Proportion of data for testing. Defaults to 0.1.
            val_size (float, optional): Proportion of data for validation. Defaults to 0.1.
            random_state (int, optional): Random seed for reproducibility. Defaults to 111.

        Returns:
            tuple: Tuple containing the training, validation, and testing DataFrames.
        """
        if sum([train_size, test_size, val_size]) != 1:
            print('Error: proportions do not add up to 1, please correct.')
        else:
            self.train_df, self.val_df = train_test_split(self.data_labeled, test_size=(test_size + val_size), random_state=random_state)
            self.val_df, self.test_df = train_test_split(self.val_df, test_size=(val_size / (val_size + test_size)), random_state=random_state)

    @classmethod
    def process_batch(cls, batch_num, train_data_generator):
        """
        Process a batch of data for augmentation.

        Args:
            batch_num (int): Batch number.
            train_data_generator (TextDataGenerator): Data generator for training data.

        Returns:
            tuple: Tuple containing augmented texts and labels.
        """
        texts, labels = train_data_generator[batch_num]
        augmented_texts_batch = train_data_generator.augment_text_batch(texts)
        return augmented_texts_batch, labels
    
    @classmethod
    def generate_augmented_data_with_progress_bar(cls, train_data_generator=None, N=0, batch_size=1, shuffle=True):
        """
        Generate augmented data with a progress bar.

        Args:
            train_data_generator (TextDataGenerator, optional): Data generator for training data. Defaults to None.
            N (int, optional): Number of augmented samples to generate. Defaults to 0.
            batch_size (int, optional): Batch size for data augmentation. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

        Returns:
            pd.DataFrame: Augmented data.
        """
        batch_size = min(N, batch_size)
        augmented_texts = []
        augmented_labels = []

        total_batches = N // batch_size + (N % batch_size != 0)

        for batch_num in tqdm(range(total_batches), total=total_batches):
            augmented_texts_batch, labels = ClassifierBERT.process_batch(batch_num, train_data_generator)
            augmented_texts.extend(augmented_texts_batch)
            augmented_labels.extend(np.atleast_1d(labels).tolist())

        augmented_df = pd.DataFrame({'text': augmented_texts, 'numeric_label': augmented_labels})
        
        return augmented_df
    
    @classmethod
    def labels_to_binary(cls, df, labels_varname, new_var_name, description_mapping):
        """
        Convert numeric labels to binary format.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            labels_varname (str): Name of the column containing the labels.
            new_var_name (str): Name of the new column to store binary labels.
            description_mapping (dict): Mapping of label descriptions.

        Returns:
            pd.DataFrame: DataFrame with binary labels.
        """
        dataset = df.copy()
        all_labels = sorted(description_mapping.keys())

        def convert_labels(row_labels):
            return [int(label in row_labels) for label in all_labels]
        
        dataset[new_var_name] = dataset[labels_varname].apply(convert_labels)
        
        return dataset
    
class PrintShapesCallback(tf.keras.callbacks.Callback):
    """
    Callback to print the shapes of the input and output batches during training.
    """
    def on_train_batch_begin(self, batch, logs=None):
        print("Input Batch Shape:", self.model.input_shape)
        print("Output Batch Shape:", self.model.output_shape)
