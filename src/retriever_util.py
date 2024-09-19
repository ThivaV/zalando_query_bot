import pandas as pd

import torch
from sentence_transformers import SentenceTransformer

from pymilvus import (
    AnnSearchRequest,
    Collection,
    WeightedRanker,
    connections
)

from langchain_milvus.utils.sparse import BM25SparseEmbedding

class InitRetriever:
    def __init__(self, device: str = 'cpu', metadata_uri: str = None, milvus_db_uri: str = None, collection_name: str = None):
        # device
        self.device = device

        # load master data
        self.metadata = pd.read_csv(metadata_uri)

        # initialize milvus db
        self.milvus_db_uri = milvus_db_uri
        self.collection_name = collection_name
        
        # connect to milvus db
        connections.connect(uri=self.milvus_db_uri)
        self.collection = Collection(self.collection_name)
        self.collection.load()

        # load sparse func
        self.bm25_model = BM25SparseEmbedding(corpus=self.metadata['description'].values.tolist())

        # load dense func
        self.clip_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=self.device)

    def search(self, query: str, sparse_weight: float = 0.5, dense_weight: float = 0.5, limit: int = 5):
        sparse_query_embeds = self.bm25_model.embed_documents([query])[0]
        dense_query_embeds = self.clip_model.encode(query)

        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest(
            [dense_query_embeds], "dense_vector", dense_search_params, limit=limit
        )

        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [sparse_query_embeds], "sparse_vector", sparse_search_params, limit=limit
        )

        rerank = WeightedRanker(sparse_weight, dense_weight)

        resp = self.collection.hybrid_search(
            [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["sku"]
        )[0]

        skus = [hit.get("sku") for hit in resp]
        filtered_df = self.metadata[self.metadata['sku'].isin(skus)]

        return filtered_df
