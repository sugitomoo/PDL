import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentence_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
sentence_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# ref https://github.com/maszhongming/MatchSum
class SiameseSentenceBERT(nn.Module):
    def __init__(self, hidden_size=384):
        super(SiameseSentenceBERT, self).__init__()
        self.hidden_size = hidden_size
        self.tokenizer = sentence_tokenizer
        self.encoder = sentence_encoder
        
    def tokenize_and_get_ids(self,lines):
        ids = []
        for line in lines:
            encoded_input = self.tokenizer.encode_plus(line, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            ids.append(encoded_input)
        return ids
    
    def forward(self, text_lines, summary_text):
        text_ids = self.tokenize_and_get_ids(text_lines)
        summary_id = self.tokenize_and_get_ids(summary_text)[0]
        summary_line_embs = []
        out = self.encoder(**summary_id)
        summary_emb = mean_pooling(out,summary_id['attention_mask']).squeeze()
        summary_line_embs.append(summary_emb)
        
        text_line_embs = []
        for text_id in text_ids:
            out = self.encoder(**text_id)
            text_emb = mean_pooling(out,text_id['attention_mask']).squeeze()
            text_line_embs.append(text_emb)
        similarities = []
        for summary_emb in summary_line_embs:
            similarity_row = []
            for text_emb in text_line_embs:
                cos_sim = torch.cosine_similarity(summary_emb,text_emb,dim = -1)
                similarity_row.append(cos_sim)
            similarities.append(torch.stack(similarity_row))
        similarities = [torch.cosine_similarity(summary_emb,text_emb,dim=-1) for text_emb in text_line_embs]
        return torch.stack(similarities)


