import pandas as pd
from PIL.ImImagePlugin import number
from sklearn.preprocessing import MultiLabelBinarizer
import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import get_close_matches



class recSysModel:
    def __init__(self, path):
        self.dataset_df = pd.read_csv(path)
        self.dataset_df["Description"] = self.dataset_df["Description"].fillna(self.dataset_df["Category"])
        self.dataset_df["Description"] = self.dataset_df["Description"].fillna(self.dataset_df["Title"])
        self.dataset_df["Category"] = self.dataset_df["Category"].fillna("")
        self.dataset_df['Category_list'] = self.dataset_df['Category'].str.split(' , ')
        self.dataset_df['Category_list'] = self.dataset_df['Category_list'].apply(lambda arr: [s.strip() for s in arr])
        mlb = MultiLabelBinarizer()
        encoded_categories = mlb.fit_transform(self.dataset_df['Category_list'])
        encoded_df = pd.DataFrame(encoded_categories, columns=mlb.classes_)
        self.dataset_df = pd.concat([self.dataset_df, encoded_df], axis=1)
        random_seed = 42
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.dataset_df = self.dataset_df.drop(columns=['Authors', 'Category', 'Category_list', 'Publisher', 'Price Starting With ($)', 'Publish Date (Month)', 'Publish Date (Year)'])
        self.titles = self.dataset_df["Title"].to_list()
    
    def __create_embedding(self, text):
        encoding = self.tokenizer.batch_encode_plus(
            [text],
            padding=True,              
            truncation=True,           
            return_tensors='pt',      
            add_special_tokens=True    
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state  
        sentence_embedding = word_embeddings.mean(dim=1)
        return sentence_embedding.cpu()
    
    def train(self, df):
        numerical_cols = self.dataset_df.select_dtypes(include=np.number).columns
        length = self.dataset_df.shape[0]
        combined_data = []
        for index, row in self.dataset_df.iterrows():
            vector = self.__create_embedding(row['Description']).reshape(768)
            numerical_values = row[numerical_cols].values
            name = row['Title']
            combined_vector = np.concatenate((vector, numerical_values))
            combined_row = np.append(name, combined_vector)
            combined_data.append(combined_row)
            print(f'Progress: {index / length:.2%}', end='\r')
        self.embeddings_df = np.array(combined_data, dtype=object)
        return self.embeddings_df
    
    def load(self, path_book_embeddings):
        self.embeddings_df = np.load(path_book_embeddings, allow_pickle = True)
    
    def predict(self, record, n=5):
        record_vector = (record[1:]).astype(np.float64).reshape(1, -1)
        record_vector = torch.from_numpy(record_vector)
        data_matrix = (self.embeddings_df[:, 1:].astype(np.float64))
        data_matrix = torch.from_numpy(data_matrix)
        all_names = self.embeddings_df[:, 0]
        distances = 1 - (torch.cosine_similiraty(record_vector, data_matrix))
        sorted_indeces = np.argsort(distances)[:n]
        names = all_names[sorted_indeces]
        return list(names)

    def predict_by_description(self, description, n=5):
        record_vector = self.__create_embedding(description)
        record_vector = torch.from_numpy(record_vector)
        data_matrix = (self.embeddings_df[:, 1:].astype(np.float64))
        data_matrix = torch.from_numpy(data_matrix)
        all_names = self.embeddings_df[:, 0]
        distances = 1 - (torch.cosine_similiraty(record_vector, data_matrix))
        sorted_indeces = np.argsort(distances)[:n]
        names = all_names[sorted_indeces]
        return list(names)

    def closest_title(self, title, size):
        results = get_close_matches(title, self.titles, n=size, cutoff=0.5)
        return results

    def get_record(self, title):
        return self.embeddings_df[self.embeddings_df[:, 0] == title]
