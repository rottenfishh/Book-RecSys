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
        self.embeddings_df = pd.DataFrame()
        numerical_cols = self.dataset_df.select_dtypes(include=np.number).columns
        length = self.dataset_df.shape[0]
        for index, row in self.dataset_df.iterrows():
            vector = self.__create_embedding(row['Description']).reshape(768)
            numerical_values = row[numerical_cols].values
            name = row['Title']
            combined_vector = np.concatenate((vector, numerical_values))
            vector_str = ','.join(map(str, combined_vector))
            new_row = pd.DataFrame({'book_embedding': [vector_str], 'name': [name]})
            self.embeddings_df = pd.concat([self.embeddings_df, new_row], ignore_index=True)
            print(f'Progress: {index / length:.2%}', end='\r')
        return self.embeddings_df
    
    def load(self, path_book_embeddings):
        self.embeddings_df = pd.read_csv(path_book_embeddings)
    
    def __parse_embedding(self, embedding_str):
        return np.array([float(x) for x in embedding_str.iloc[0].split(',')])

    def __parse_emb_str(self, embedding_str):
        return np.array([float(x) for x in embedding_str.split(',')])

    def predict(self, record, n=5):
        record_vector = self.__parse_embedding(record['book_embedding']).reshape(1, -1)
        distances = []
        names = []
        length = self.embeddings_df.shape[0]
        for index, row in self.embeddings_df.iterrows():
            other_vector = self.__parse_emb_str(row['book_embedding']).reshape(1, -1)
            similarity = cosine_similarity(record_vector, other_vector)[0][0]
            distance = 1 - similarity
            distances.append(distance)
            names.append(row['name'])
            print(f'Progress: {index / length:.2%}', end='\r')
        sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
        closest_names = [names[i] for i in sorted_indices[:n]]
        return closest_names

    def predict_by_description(self, description, n=5):
        record_vector = self.__create_embedding(description)
        distances = []
        names = []
        length = self.embeddings_df.shape[0]
        for index, row in self.embeddings_df.iterrows():
            other_vector = self.__parse_emb_str(row['book_embedding']).reshape(1, -1)
            other_vector = other_vector[:, :768]
            similarity = cosine_similarity(record_vector, other_vector)[0][0]
            distance = 1 - similarity
            distances.append(distance)
            names.append(row['name'])
            print(f'Progress: {index / length:.2%}', end='\r')
        sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
        closest_names = [names[i] for i in sorted_indices[:n]]
        return closest_names

    def closest_title(self, title, size):
        results = get_close_matches(title, self.titles, n=size, cutoff=0.5)
        return results

    def get_record(self, title):
        return self.embeddings_df[self.embeddings_df['name'] == title]



