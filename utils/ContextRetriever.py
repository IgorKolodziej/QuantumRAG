import json
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import requests
import os
import traceback
import sys
from tqdm.auto import tqdm
#importutils which are ../../utils/ContextRetrieverUtils.py
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# from ..utils.ContextRetrieverUtils import ContextRetrieverUtils
from utils.ContextRetrieverUtils import ContextRetrieverUtils

class ContextRetriever:
    def __init__(self, model_name, embeddings_file=None, docs_list_file=None, device=None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        except OSError as e:
            print(f"Error loading model '{self.model_name}': {e}")
            raise

        self.index = None
        self.documents = []
        self.embeddings_file = embeddings_file
        self.docs_list_file = docs_list_file

    def build_index(self, documents_to_index, batch_size=32):
        if not documents_to_index:
            print("Warning: No documents provided to build_index. Index will be empty.")
            self.documents = []
            embedding_dim = self.model.config.hidden_size
            self.index = faiss.IndexFlatIP(embedding_dim)
            print(f"FAISS index initialized empty.")
            return

        doc_embeddings = None
        loaded_from_file = False

        if self.embeddings_file and self.docs_list_file and \
                os.path.exists(self.embeddings_file) and os.path.exists(self.docs_list_file):
            print(f"Attempting to load documents ({self.docs_list_file}) and embeddings ({self.embeddings_file})...")
            try:
                with open(self.docs_list_file, 'r', encoding='utf-8') as f:
                    loaded_docs_list = json.load(f)

                if loaded_docs_list == documents_to_index:
                    print("Document lists match. Loading embeddings...")
                    loaded_embeddings = np.load(self.embeddings_file)
                    if loaded_embeddings.shape[0] == len(documents_to_index):
                        print(
                            f"Successfully loaded {loaded_embeddings.shape[0]} embeddings from {self.embeddings_file}")
                        doc_embeddings = loaded_embeddings
                        self.documents = documents_to_index
                        loaded_from_file = True
                    else:
                        print(f"Embeddings shape mismatch. Re-generating.")
                else:
                    print("Document lists do NOT match. Re-generating embeddings.")
            except Exception as e:
                print(f"Error loading from file or validating: {e}. Re-generating embeddings.")

        if not loaded_from_file:
            print(f"Generating document embeddings using '{self.model_name}' with 'CLS' pooling...")
            self.documents = documents_to_index
            doc_embeddings = ContextRetrieverUtils.get_embeddings(
                self.documents, self.model, self.tokenizer, self.device, batch_size=batch_size)
            if self.embeddings_file and self.docs_list_file:
                print(f"Saving embeddings to {self.embeddings_file} and docs list to {self.docs_list_file}")
                try:
                    if os.path.dirname(self.embeddings_file): os.makedirs(os.path.dirname(self.embeddings_file),
                                                                          exist_ok=True)
                    if os.path.dirname(self.docs_list_file): os.makedirs(os.path.dirname(self.docs_list_file),
                                                                         exist_ok=True)
                    np.save(self.embeddings_file, doc_embeddings)
                    with open(self.docs_list_file, 'w', encoding='utf-8') as f:
                        json.dump(self.documents, f, ensure_ascii=False, indent=2)
                    print("Save complete.")
                except Exception as e:
                    print(f"Error saving embeddings or docs list: {e}")

        if doc_embeddings is None:
            print("Error: Document embeddings are None.")
            return

        print("Normalizing document embeddings for FAISS IndexFlatIP (cosine similarity)...")
        faiss.normalize_L2(doc_embeddings)
        embedding_dim = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(doc_embeddings)
        print(f"FAISS index built with {self.index.ntotal} documents.")

    def search(self, query_text_with_prefix, top_k=10):
        if self.index is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")
        if self.index.ntotal == 0:
            return []

        query_embedding = ContextRetrieverUtils.get_embeddings([query_text_with_prefix], self.model, self.tokenizer,
                                         self.device, batch_size=1)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for i in range(len(indices[0])):
            doc_index = indices[0][i]
            if 0 <= doc_index < len(self.documents):
                dist = distances[0][i]
                results.append({
                    "document": self.documents[doc_index],
                    "similarity_score": float(dist),
                    "rank": i + 1
                })
        return results

if __name__ == "__main__":
    model_name = 'mixedbread-ai/mxbai-embed-large-v1'
    # Required prefix for mxbai-embed-large-v1 retrieval queries
    query_prefix = "Represent this sentence for searching relevant passages: "
    squad_url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'
    squad_filename = 'train-v2.0.json'
    dataset_dir = "squad_dataset"
    embeddings_dir = "saved_embeddings"
    model_slug = model_name.replace("/", "_").replace("-", "_")
    squad_filepath = os.path.join(dataset_dir, squad_filename)
    saved_embeddings_path = os.path.join(embeddings_dir, f"squad_embeddings_{model_slug}.npy")
    saved_docs_list_path = os.path.join(embeddings_dir, f"squad_docs_{model_slug}.json")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    if not ContextRetrieverUtils.download_squad(squad_filepath, squad_url):
        print("Failed to download or access SQuAD dataset. Exiting.")
        exit()

    print("Loading SQuAD documents...")
    all_squad_documents = ContextRetrieverUtils.load_squad_contexts(squad_filepath)
    documents_to_process = all_squad_documents

    print(f"Processing {len(documents_to_process)} unique context documents for model '{model_name}'.")

    if not documents_to_process:
        print("No documents loaded from SQuAD file. Exiting.")
        exit()

    try:
        retriever = ContextRetriever(
            model_name=model_name,
            embeddings_file=saved_embeddings_path,
            docs_list_file=saved_docs_list_path
        )
        retriever.build_index(documents_to_process, batch_size=32)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        exit()

    test_queries = [
        "Who was the first president of the United States?",
        "What is the process of photosynthesis in plants?",
        "What is the main function of the mitochondria?",
        "Who wrote 'Pan Tadeusz'?",
        "What are the implications of quantum entanglement?",
        "What are the main causes of climate change?",
        "Can you explain the theory of relativity?"
    ]

    for raw_test_query in test_queries:
        print(f"\n-----\nUser Query: '{raw_test_query}'")

        query_for_model = f"{query_prefix}{raw_test_query}"

        print(f"Query sent to model: '{query_for_model}'")
        search_results = retriever.search(query_for_model, top_k=5)

        if search_results:
            print("Top Retrieved Documents:")
            for result in search_results:
                print(f"Rank {result['rank']} (Score: {result['similarity_score']:.4f})")
                print(f"{result['document'][:350]}...\n")
        else:
            print("No results found for this query.")
