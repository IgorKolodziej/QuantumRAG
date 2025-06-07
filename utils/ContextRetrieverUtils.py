import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import requests
import os
import traceback
from tqdm.auto import tqdm
class ContextRetrieverUtils:
    @staticmethod
    def download_squad(filepath, url):
        if not os.path.exists(filepath):
            print(f"SQuAD dataset not found at {filepath}.")
            print(f"Downloading SQuAD dataset from {url} to {filepath}...")
            dirname = os.path.dirname(filepath)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                with open(filepath, 'wb') as file, tqdm(
                        desc=os.path.basename(filepath), total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                        bar.update(len(chunk))
                print("Download complete.")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading SQuAD dataset: {e}")
                if os.path.exists(filepath): os.remove(filepath)
                return False
        else:
            print(f"SQuAD dataset already exists at {filepath}")
            pass
        return True
    @staticmethod
    def load_squad_contexts(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            squad_data = json.load(f)

        contexts = set()
        for article in squad_data['data']:
            for paragraph in article['paragraphs']:
                contexts.add(paragraph['context'])
        return list(contexts)

    @staticmethod
    def get_embeddings(texts, model, tokenizer, device, batch_size=32):
        all_embeddings = []
        model.eval()
        effective_max_length = min(512, tokenizer.model_max_length)
        print(f"Using effective max_length: {effective_max_length}, pooling: CLS")

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", leave=False):
                batch_texts = texts[i:i + batch_size]
                encoded_input = tokenizer(
                    batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=effective_max_length
                ).to(device)
                model_output = model(**encoded_input)
                sentence_embeddings = model_output.last_hidden_state[:, 0] # CLS pooling

                all_embeddings.append(sentence_embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)
