# Text Classification Using BERT Model

## Overview

This package provides a solution for training and evaluating a text classification model based on the BERT architecture (Bidirectional Encoder Representations from Transformers by Google). The package processes input data, deduplicates repeated entries, trains a model on labeled data, and makes predictions on unlabeled data. This README provides a description of the main functions and classes, along with examples of how to use them.

In technical terms, it uses a model called BERT (Bidirectional Encoder Representations from Transformers). The BERT model has been previously trained on vast amounts of text data, giving it a good understanding of language and context. It can read a sentence and understand the context in which each word is used. For instance, it knows that in the sentence "Apple is delicious", "Apple" refers to the fruit, but in "Apple released a new phone", it refers to the company.

## Requirements

- pandas
- TensorFlow
- transformers
- tensorflow_addons
- scikit-learn
- numpy
- matplotlib

## 1. `HTML_folder_parser`

This module provides functionalities to parse HTML files from a given folder.

- **Loading Data**: 
    - The data is loaded from a specific folder using the `HTML_folder_parser` function. This function reads and parses the HTML files present in the specified folder.
- **Excluding Unwanted Data**:
    - Some file types, like images (`jpg`, `png`) or Word documents (`docx`), might not be immediately usable for text classification. These are excluded using the `exclude_values` variable.
- **Data Cleaning**:
    - The code checks for any entries that contain names and email using regex and replace them with `[EMAIL REMOVED]`, `[NAME REMOVED]`, or `[ADDRESS REMOVED]` to focus on the content of the text instead of the author.
- **Processing HTML and PDF Files**:
    - The module can handle both HTML and PDF files. It uses BeautifulSoup to parse HTML files and PDFMiner to extract text from PDF files.
- **Progress Display**:
    - While parsing the files, the module provides a progress display to keep track of the processing status.
 
**Usage:**

```python
import HTML_folder_parser

# folder_path correspondes to the folder containing all html files.
# project_prefix contains the prefix of the files that will be processed within the folder.
MyParser = HTML_folder_parser(folder_path, project_prefix)
unique_files = MyParser.find_files_and_attachments()
unique_entries = MyParser.read_and_parse_html(unique_files)
```

## 2. `DeDuplication`
This module contains the `RecordLinkage` class that helps in deduplicating text data by finding fuzzy string matches.
- **Finding Repetitive Text**:
    - The `RecordLinkage` class is used to compare texts and identify those that are very similar or even identical.
    - The method compares words and n-grams (pieces of text of length "n") using cosine similarity based on TF-IDF (relative term frequency across documents, inverted). Essentially, it compares vector representations of phrase segments, giving more weight to infrequent patterns.
    - The `deduplication` method in the `RecordLinkage` class removes these duplicates, ensuring the model doesn't waste time on repetitive content.
- **Grouping Similar Texts**:
    - After deduplication, the texts that are similar are grouped together using the `generate_group_ids method`, allowing for efficient processing of grouped texts.

**Usage:**

```python
import RecordLinkage

DeDuplicator = RecordLinkage(unique_entries, 'comment', n_gramsize=[7])
deduplicated = DeDuplicator.deduplication(ntop=75000, lower_bound=0.65, max_matches=None, n_grams=[8], full_words=True)
new_group_ids = DeDuplicator.generate_group_ids(deduplicated)
```

**Parameters explained:**
- `unique_entries`: A DataFrame containing the entries to be deduplicated. Each row should have the text data to be compared.
- `messy_names_var`: The name of the column in the DataFrame containing the text data to be deduplicated.
- `n_gramsize`: A list of integers specifying the sizes of n-grams (substrings of length "n") used for comparing texts.
- `ntop`: The number of top matches to consider for each entry (default: 5).
- `lower_bound`: The lower bound similarity threshold for matches (default: 0.8). Only pairs with similarity scores above this threshold are considered matches.
- `max_matches`: The maximum number of top matches to include in the output DataFrame (default: 1000).
- `n_grams`: A list of integers specifying the sizes of n-grams to be used in the TF-IDF vectorization.
- `full_words`: A boolean indicating whether to include full words in the n-grams list for comparison (default: True).
  
## 3. `ClassifierBERT`

This is the main class for training and evaluating a BERT-based text classification model.

1. **Tokenization**:
    - Each piece of text is broken down into smaller chunks called tokens. For example, "I love apples" might be broken down into "I", "love", and "apples", which are then converted into numeric representations (large vectors) that the model can understand.

2. **Data Augmentation**: Data augmentation involves randomly modifying the text data while preserving its meaning. This helps in making the model more robust. The methods used include:
   - Backtranslation: Translating the text into another language and then back to the original language using the MarianMT model.
   - Synonym replacement: Randomly replacing a few words in the text with their synonyms.
   - Random insertion: Randomly inserting synonyms into the text.
   - Random deletion: Randomly deleting some words from the text.

4. **Feeding Data to the Model**:
    - Once the text data is tokenized, it is fed into the BERT model. The model reads these tokens and tries to predict the correct category for each text.

5. **Adjusting Based on Mistakes**:
    - Initially, the model will make mistakes. However, it learns and improves from these mistakes by adjusting its parameters during training. This process is akin to learning from errors.

6. **Validation**:
    - To ensure the model is learning correctly, it is periodically tested with a validation dataset (data it hasn't seen before but for which the correct answers are known). This step ensures the model is generalizing well and not just memorizing the training data.

7. **Completion**:
    - After training for a specified number of epochs (iterations over the training dataset), the model becomes capable of efficiently reading and categorizing new texts.

**Usage:**

### Instantiation:

```python
MyClassifier = ClassifierBERT(labeled=model_data, text_varname='text_column', labels_varname='label_column', model='distilbert-base-uncased')
```

**Parameters explained:**
- `labeled (pd.DataFrame)`: The DataFrame containing the labeled data for training the model. Each row should include text data and corresponding labels.
- `text_varname (str)`: The name of the column in the DataFrame that contains the text data.
- `labels_varname (str)`: The name of the column in the DataFrame that contains the labels.
- `model (str, optional)`: The pretrained BERT model to use. Defaults to 'distilbert-base-uncased'. Options include 'bert-base-uncased' and 'distilbert-base-uncased'.

### Training:

```python
MyClassifier.train_model(epochs=30, batch_size=16, train_size=0.7, test_size=0.15, val_size=0.15,
                         initial_learning_rate=3e-5, data_augmentation=True, N=int(len(model_data)*0.5))
```
**Parameters explained:**
- `epochs (int, optional)`: Number of training epochs. Defaults to 2.
- `batch_size (int, optional)`: Batch size for training. Defaults to 32.
- `train_size (float, optional)`: Proportion of data for training. Defaults to 0.8.
- `test_size (float, optional)`: Proportion of data for testing. Defaults to 0.1.
- `val_size (float, optional)`: Proportion of data for validation. Defaults to 0.1.
- `initial_learning_rate (float, optional)`: Initial learning rate. Defaults to 2e-5.
- `data_augmentation (bool, optional)`: Whether to apply data augmentation. Defaults to False.
- `N (int, optional)`: Number of augmented samples to generate. Defaults to 0.

### Predicting:

```python
unlabeled_data_unique = unlabeled_data[['comment','group_id']].drop_duplicates(subset='group_id', keep='first', inplace=False)
df_human_readable_labels = MyClassifier.predict_unlabeled_data(unlabeled_data_unique, 'comment')
```
**Parameters explained:**
- `unlabeled_data_unique (pd.DataFrame)`: DataFrame with unique unlabeled data.
- `unlabeled_text_varname (str)`: Name of the column containing the text data.
