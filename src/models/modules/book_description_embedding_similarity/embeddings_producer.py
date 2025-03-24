import torch
from transformers import BertTokenizer, BertModel

class EmbeddingsProducer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def create_embedding(self, text):
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
