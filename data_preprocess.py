from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from PYPDF2 import PdfFileReader
import io
import csv
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "mssql+pyodbc://username:password@server/database"
db = SQLAlchemy(app)

class DataProcessor:
    def __init__(self, file_path):
        self.app = None
        self.file_path = file_path
        self.data = None

    def get_data_from_sql_server(self):
        # Connect to the SQL Server database
        query = "SELECT * FROM your_table" # change to my database table
        self.data = db.engine.execute(query).fetchall()
    
    def read_pdf_file(self, file_path):
        #open the pdf file in read-binary mode
        with open(file_path, 'rb') as file:
            pdf_reader = PdfFileReader(file)
            num_pages = pdf_reader.numPages
            text = ''
            for page in range(num_pages):
                page_obj = pdf_reader.getPage(page)
                text += page_obj.extractText()
            return text

    def extract_text_from_pdf(self, file_path):
        # Open the pdf file in read-binary mode
        with open(file_path, 'rb') as file:
            # Create a PDF parser object
            parser = PDFParser(file)
            #Create a pdf document object
            doc = PDFDocument(parser)
            #Create a PDF resource manager object
            rsrcmgr = PDFResourceManager()
            #create a string buffer for the extracted text
            output_string = io.StringIO()
            #create a text converter Object
            device = TextConverter(rsrcmgr, output_string, laparams= LAParams())
            #create a PDF page interpreter object
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            #Iterate through the pages and extract the text 
            for page in PDFPage.get_pages(file):
                interpreter.process_page(page)
            return output_string.getvalue()
    
    def read_csv_file(self):
        with open(self.file_path, 'r') as file:
            reader = csv.reader(file)
            self.data = list(reader)
    
    def extract_data(self):
        self.read_csv_file()
        return self.data
    
    def extract_data_to_pandas(self):
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def get_column_names(self):
        self.extract_data_to_pandas()
        return list(self.data.columns)
    
    def get_column_data(self, column_name):
        self.extract_data_to_pandas()
        return self.data.iloc[column_name].tolist()
    
    def get_row_data(self, row_index):
        self.extract_data_to_pandas()
        return self.data[row_index].tolist()
    
    def get_cell_data(self, row_index, column_index):
        self.extract_data_to_pandas()
        return self.data.iloc[row_index, column_index]
    
    def filter_data(self, column_name, filter_value):
        self.extract_data_to_pandas()
        return self.data[self.data[column_name] == filter_value]
    
    def sort_data(self, column_name, ascending=True):
        self.extract_data_to_pandas()
        return self.data.sort_values(by=column_name, ascending=ascending)
    
    def group_data(self, column_name):
        self.extract_data_to_pandas()
        return self.data.groupby(column_name)
    
    def aggregate_data(self, column_name, aggregation_function):
        self.extract_data_to_pandas()
        return self.data.groupby(column_name).agg(aggregation_function)
    
    def process_data(self, file_path):
        text = self.extract_text_from_pdf(file_path)
        #process the extracted text into a readable format
        #split the text into lines and remove any unnecessary characters
        lines = text.split('\n')
        processed_text = [line.strip() for line in lines]
        return processed_text

"""
extractor = CSVDataExtractor('example.csv')
data = extractor.extract_data()
print(data)

column_names = extractor.get_column_names()
print(column_names)

column_data = extractor.get_column_data('column_name')
print(column_data)

row_data = extractor.get_row_data(0)
print(row_data)

cell_data = extractor.get_cell_data(0, 0)
print(cell_data)

filtered_data = extractor.filter_data('column_name', 'filter_value')
print(filtered_data)

sorted_data = extractor.sort_data('column_name')
print(sorted_data)

grouped_data = extractor.group_data('column_name')
print(grouped_data)

aggregated_data = extractor.aggregate_data('column_name', 'sum')
print(aggregated_data)

"""
