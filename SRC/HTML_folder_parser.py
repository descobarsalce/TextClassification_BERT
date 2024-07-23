# -*- coding: utf-8 -*-
"""
Created on Wed Oct 1 2023
Edited on Mon Jul 22 2024
@author: DEscobar-Salce
"""

from bs4 import BeautifulSoup
import os
import regex as re
import pandas as pd
from pdfminer.high_level import extract_text
import spacy  # if not downloaded, use the following: python -m spacy download en_core_web_sm

class HTML_folder_parser():
    """
    A class to parse HTML and PDF files from a given folder for text classification tasks.
    
    Attributes:
    folder_path : str
        The path to the folder containing the files to be parsed.
    project_prefix : str
        The common prefix for the project files.
    clean_data : bool
        Flag to indicate whether to clean personal information from the data.
    nlp : spacy.language.Language
        spaCy's language model for named entity recognition (NER).
    """
    
    def __init__(self, folder_path, project_prefix, clean_data=True):
        self.folder_path = folder_path
        self.project_prefix = project_prefix
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy's English NER model
        self.clean_data = clean_data
        
    def parse_conditionally(self, row):
        """
        Process different types of files' data based on their extension.
        
        Parameters:
        row : pd.Series
            A row from the dataframe containing file information.
        
        Returns:
        str or None
            Parsed text from the file or None if an error occurs.
        """
        try:
            if row['extension'] == 'html':
                return self.parse_html_file(os.path.join(self.folder_path, row['file_name']))
            elif row['extension'] == 'pdf':
                return self.parse_PDF_file(os.path.join(self.folder_path, row['file_name']))
        except:
            return None  # or some default value if not 'html'
    
    def progress_display(self, row, **kwargs):
        """
        Display progress during HTML/PDF parsing.
        
        Parameters:
        row : pd.Series
            A row from the dataframe containing file information.
        kwargs : dict
            Additional arguments including total rows and progress interval.
        
        Returns:
        str or None
            Parsed text from the file or None if an error occurs.
        """
        index = row.name  # Get the index of the row
        total_rows = kwargs.get('total_rows')
        progress_interval = kwargs.get('progress_interval', 100)
        if index % progress_interval == 0:
            print(f"\rProcessing HTML/PDF: {index}/{total_rows} rows...", end='', flush=True)
        return self.parse_conditionally(row)

    @staticmethod
    def find_comment(parsed_html):
        """
        Extract the 'General Comment' from parsed HTML.
        
        Parameters:
        parsed_html : dict or str
            Parsed HTML content.
        
        Returns:
        str
            General comment text.
        """
        if type(parsed_html) == dict:
            return parsed_html['General Comment']
        elif type(parsed_html) == str:
            return parsed_html

    def read_and_parse_html(self, unique_files):
        """
        Read and parse HTML files, displaying progress during processing.
        
        Parameters:
        unique_files : pd.DataFrame
            DataFrame containing file information.
        
        Returns:
        pd.DataFrame
            DataFrame with parsed HTML content.
        """
        total_rows = len(unique_files)
        progress_interval = max(total_rows // 100, 1)  # Update every 1% for example
        kwargs = {'total_rows': total_rows, 'progress_interval': progress_interval}
        unique_files['parsed_html'] = unique_files.apply(
            lambda row: self.progress_display(row, **kwargs), axis=1)
        unique_files['comment'] = unique_files['parsed_html'].apply(lambda x: self.find_comment(x))
        return unique_files
    
    @classmethod
    def clean_personal_info(cls, row):
        """
        Clean personal information from the comments in the parsed HTML.
        
        Parameters:
        row : pd.Series
            A row from the dataframe containing parsed HTML content.
        
        Returns:
        str
            Cleaned comment text.
        """
        comment = row['comment']
        if type(row['parsed_html']) == dict:
            if row['parsed_html'].get('Email:'):
                comment = comment.replace(row['parsed_html'].get('Email:'), "[EMAIL REMOVED]")
            if row['parsed_html'].get('Name:'):
                comment = comment.replace(row['parsed_html'].get('Name:'), "[NAME REMOVED]")    
        comment = HTML_folder_parser.remove_personal_info(comment)
        return comment
        
    @staticmethod
    def remove_personal_info(text):
        """
        Remove personal information such as emails and addresses from text.
        
        Parameters:
        text : str
            Text from which personal information needs to be removed.
        
        Returns:
        str
            Text with personal information removed.
        """
        # Define the email pattern
        email_pattern = r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b'
        # Remove emails
        text = re.sub(email_pattern, "[EMAIL REMOVED]", text)
        # Find all potential addresses (broad pattern to catch candidates)
        potential_addresses = re.findall(r'\b\d{1,5}\b[-\s\w]+', text)
        # Filter the found addresses, checking them with a more specific pattern
        for potential_address in potential_addresses:
            if HTML_folder_parser.looks_like_us_address(potential_address):
                text = text.replace(potential_address, "[ADDRESS REMOVED]")
        return text
    
    @staticmethod
    def looks_like_us_address(text):
        """
        Check if the text looks like a US address.
        
        Parameters:
        text : str
            Text to be checked.
        
        Returns:
        bool
            True if the text looks like a US address, False otherwise.
        """
        us_address_pattern = (
            r'\b(?:N\.?|S\.?|E\.?|W\.?)?\s*\d{1,5}\b\s'  # Optional direction and street number
            r'(?:[\w.]+\s){0,4}'  # Street name (up to four words)
            r'(?:Avenue|Ave\.?|Street|St\.?|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Court|Ct\.?|Lane|Ln|Terrace|Ter\.?|Place|Pl|Circle|Cir\.?)\s'  # Street type
            r'(?:[\w.]+\s){0,2}(?:,\s)?'  # Optional city (up to two words) with an optional comma
            r'(?:[A-Z]{2}\s)?'  # Optional state abbreviation
            r'\d{5}'  # ZIP code
        )
        return bool(re.search(us_address_pattern, text, re.IGNORECASE))  # Case-insensitive match

    def parse_html_file(self, file_path):
        """
        Parse an HTML file.
        
        Parameters:
        file_path : str
            Path to the HTML file.
        
        Returns:
        dict
            Parsed content of the HTML file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
            return self.parse_html(html_content)
        
    def parse_PDF_file(self, file_path):
        """
        Parse a PDF file.
        
        Parameters:
        file_path : str
            Path to the PDF file.
        
        Returns:
        str
            Extracted text from the PDF file.
        """
        return self.extract_text_from_pdf_LLM(file_path)

    def parse_html(self, html_content, starting_texts=["Docket:", "Comment On:", "Document:", "Name:", "Email:"]):
        """
        Parse the HTML content and extract relevant information.
        
        Parameters:
        html_content : str
            Raw HTML content.
        starting_texts : list
            List of starting texts to look for specific elements.
        
        Returns:
        dict
            Parsed data from the HTML content.
        """
        # Initializing the BeautifulSoup object
        soup = BeautifulSoup(html_content, 'html.parser')
        data = {}
        
        # Extracting the table with "PUBLIC SUBMISSION" header
        data['table_metadata'] = soup.find('h1', string='PUBLIC SUBMISSION').find_parent('table').get_text(strip=True, separator=' ')
        
        # Extracting the elements with specific starting text
        for text in starting_texts:
            element = soup.find(string=lambda x: x and x.startswith(text))
            if element:
                content = element.find_next(string=True).strip()  # Get the next string after the element
                data[text] = content
        
        # Extracting the General Comment paragraph
        general_comment_header = soup.find('h2', class_='aligncenter debug', string='General Comment')
        if general_comment_header:
            general_comment_para = general_comment_header.find_next_sibling('p', class_='alignleft debug')
            data['General Comment'] = general_comment_para.get_text(strip=True)
        return data
    
    @staticmethod
    def extract_text_from_pdf_LLM(pdf_path):
        """
        Extract text from a PDF file using PDFMiner.
        
        Parameters:
        pdf_path : str
            Path to the PDF file.
        
        Returns:
        str
            Extracted text from the PDF file.
        """
        text = extract_text(pdf_path)
        # Remove common header/footer elements. This can be tailored based on the specific patterns in the PDFs.
        text = re.sub(r'Page \d+ of \d+', '', text)  # Remove "Page x of y" patterns
        text = re.sub(r'\n\d+\n', '\n', text)  # Remove isolated page numbers
        # Handle double spacing by replacing multiple consecutive newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        # Join broken sentences (this handles lines that end with a hyphen to indicate a word break)
        text = re.sub(r'-\n', '', text)
        return text

    @staticmethod
    def is_footnote(line):
        """
        Check if a line is likely a footnote based on heuristics.
        
        Parameters:
        line : str
            Line of text to be checked.
        
        Returns:
        bool
            True if the line is likely a footnote, False otherwise.
        """
        # Check for single digit followed by a period (e.g., "1.")
        if re.match(r'^\d\.$', line.strip()):
            return True
        # Check for short lines (e.g., less than 20 characters) starting with a digit
        if len(line) < 20 and line[0].isdigit():
            return True
        return False

    def join_continuation_lines(self, text):
        """
        Join continuation lines in the extracted text.
        
        Parameters:
        text : str
            Extracted text from a PDF file.
        
        Returns:
        list
            Processed lines of text with joined continuations.
        """
        # Split the text into lines
        lines = text.split('\n')
        
        # Initialize an empty list to hold the processed lines
        processed_lines = []

        # Run over all paragraphs    
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Conditions for joining lines:
            # 1. Current line doesn't end with specific punctuation
            # 2. Current line is longer than 60 characters
            # 3. Next line doesn't start with a number indicative of a footnote
            while (i < len(lines) - 1) and \
                  (not line.endswith(('.', '?', '!', ';', ':', '-', '—'))) and \
                  len(line) > 60 and \
                  (not self.is_footnote(lines[i + 1])):
                      
                i += 1  # Move to next line
                line += " " + lines[i].strip()  # Append next line to current line
            
            processed_lines.append(line)
            i += 1  # Move to next line
            
        processed_lines.append(line)
        
        # Join the processed lines back into a single string
        return processed_lines

    @staticmethod
    def parse_suffix(file_name):
        """
        Parse the suffix of a file name.
        
        Parameters:
        file_name : str
            The name of the file.
        
        Returns:
        tuple
            Parsed number, suffix, and extension from the file name.
        """
        match = re.match(r'(\d+)(?:-([^-.]+))?(\..+)?', file_name)
        if match:
            number = match.group(1)
            suffix = match.group(2) if match.group(2) else ""
            extension = match.group(3) if match.group(3) else ""
    
            # Parse extension into a different bin
            if extension:
                extension_bin = extension[1:]  # Exclude the dot from the extension
            else:
                extension_bin = None
    
            return number, suffix, extension_bin
        else:
            return None
    
    def find_files_and_attachments(self):
        """
        Find files and attachments in the specified folder.
        
        Returns:
        pd.DataFrame
            DataFrame with information about the found files and attachments.
        """
        if os.path.exists(self.folder_path):    
            # List all files in the folder
            files = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]
        unique_entries = {}
        
        progress_update_freq = 100
        total = len(files)
        bar_length = 50
        
        # Group files by what comes after the common prefix
        for counter, file in enumerate(files):
            if file.startswith(self.project_prefix):
                suffix = file[len(self.project_prefix):].lstrip('-').lstrip('.')
            if suffix:            
                match = re.match(r'(\d+)(?:-([^.-]+))?(?:\.(.+))?$', suffix)
                if match:
                    number = match.group(1)
                    suffix = match.group(2) if match.group(2) else ""
                    extension = match.group(3) if match.group(3) else ""
                if number:
                    file_data = {}
                    file_data['file_name'] = file
                    file_data['extension'] = extension
                    # The following combines entries into a dictionary whenever there is more than one entry for a comment number.
                    unique_entries[number] = unique_entries.get(number, []) + [file_data]
            if counter % progress_update_freq == 0:
                progress = (counter / total)
                arrow = '=' * int(round(progress * bar_length) - 1) + '>'
                spaces = ' ' * (bar_length - len(arrow))
                # This will print the progress bar and overwrite the previous line
                print(f"\rProcessing Files List: [{arrow + spaces}] {int(progress * 100)}%", end='', flush=True)    
        print("")
            
        # Transform the dictionary
        transformed_data = []
        for key, file_list in unique_entries.items():
            for file_info in file_list:
                transformed_data.append(
                    {'double_key': f"{key}_{file_info['extension']}",
                     'entry_code': key,
                     'extension': file_info['extension'],
                     'file_name': file_info['file_name']},
                    )
                
        # Convert to dataframe
        df = pd.DataFrame(transformed_data)
        
        return df
                
    @classmethod
    def find_attachments_exclusions_NA(cls, df, drop_drafts=False):
        """
        Find attachments and exclude certain files based on criteria.
        
        Parameters:
        df : pd.DataFrame
            DataFrame containing file information.
        drop_drafts : bool, optional
            Flag to indicate whether to drop draft files (default is False).
        
        Returns:
        pd.DataFrame
            DataFrame with attachments and exclusions processed.
        """
        # Delete drafts loaded from the main folder
        if drop_drafts:
            pattern = r'(?i)(DRAFT)'
            df.contains_pattern = df['file_name'].str.contains(pattern, na=False)
            df = df[df.contains_pattern == False]

        # Find files with attachments by checking whether there is more than one file with the same ID
        df['has_attachments'] = (df.groupby('entry_code')['entry_code'].transform('count')) > 1
        df['has_attachments'].value_counts()
        
        # Filter those values that this is not processing yet
        exclude_values = ['jpg', 'png', 'docx']  # CHECK!
        df['excluded_extensions'] = df['extension'].isin(exclude_values)
        
        # Perform some data cleaning/homogenization
        df['NA_values'] = df.comment.isna() == True
        
        # The following observations are excluded from deduplication
        excluded_sample = (df.excluded_extensions) | (df.NA_values)
        print(excluded_sample.value_counts())
        
        df['attached_text'] = df['comment'].str.contains('attach', case=False, na=False)
        df['short_text'] = df['comment'].str.len() < 50
        df = df[~(df['has_attachments'] & df['attached_text'] & df['short_text'])]

        df = df[~excluded_sample]
        
        return df
