import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import (
    BaseRepresentation,
    MaximalMarginalRelevance,
)
from hdbscan import HDBSCAN
# from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap.umap_ import UMAP
from model2vec import StaticModel

class SenTopic:
    """SenTopic combines the BERTopic topic modeling technique with sentiment analysis."""

    def __init__(self, 
                 embedding_model: str =  None, 
                 umap_model: UMAP =  None, 
                 hdbscan_model: HDBSCAN = None, 
                 vectorizer_model: CountVectorizer = None, 
                 representation_model: BaseRepresentation = None,
                 sentiment_model: str = None):
        """SenTopic.

        :param embedding_model: string specifiying the embedding model name.
                                SentenceTransformer models supported only. 
                                The default is 'all-MiniLM-L6-v2'
        :param umap_model: pass in a UMAP or use the default.
        :param hdbscan_model: pass in a HDBSCAN model or use the default.
        :param vectorizer_model: pass in a CountVectorizer model or use the default.
        :param representation_model: pass in a BaseRepresentation model or use the default.
        :param sentiment_model: pass in a sentiment model from hugging face or use the default.
        """
        ## default settings for embedding, umap, hdbscan, vectorizer, and representation 
        if embedding_model is None:
            self.embedding_model = StaticModel.from_pretrained("minishlab/potion-base-8M")
        # elif isinstance(embedding_model, str):
        #     self.embedding_model = SentenceTransformer(embedding_model)

        if umap_model is not None:
            self.umap_model = umap_model
        else:
            self.umap_model = UMAP(
                n_neighbors= 4,
                n_components=16,
                min_dist = 0.1,
                metric = 'cosine',
            ) 
        
        if hdbscan_model is not None:
            self.hdbscan_model = hdbscan_model
        else:
            self.hdbscan_model=HDBSCAN(
                min_cluster_size=5,
                metric='euclidean',
                prediction_data=False,
            )
        
        self.vectorizer_model = vectorizer_model or CountVectorizer(stop_words="english")
        self.representation_model = representation_model or MaximalMarginalRelevance(diversity = 0.2)

        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            representation_model=self.representation_model
        )

    def fit_transform(self, 
                      texts: list[str],
                      embeddings: np.ndarray = None,
                      ) -> tuple[list[int], np.ndarray | None]:
        return self.topic_model.fit_transform(texts, embeddings)
        

    def visualize_documents(self, texts, embeddings):
        return self.topic_model.visualize_documents(texts, embeddings = embeddings)

    def get_topic_info(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = df['chunks'].tolist()
        review_ids = df['review_ids'].tolist()
        chunk_ids = df['chunk_ids'].tolist()
        positive = [1 if sentiment == 'positive' else 0 for sentiment in df['sentiment'].tolist()]
        negative = [1 if sentiment == 'negative' else 0 for sentiment in df['sentiment'].tolist()]
        neutral = [1 if sentiment == 'neutral' else 0 for sentiment in df['sentiment'].tolist()]

        topic_df = self.topic_model.get_document_info(texts)
        topic_df['text'] = texts
        topic_df['review_id'] = review_ids
        topic_df['chunk_id'] = chunk_ids
        topic_df['positive'] = positive
        topic_df['negative'] = negative
        topic_df['neutral'] = neutral
        grouped = topic_df.groupby(by = ['Topic'])

        final_df = grouped[
            ["Name", 
             "Top_n_words",
            'review_id', 
            'chunk_id',
            'positive', 
            'neutral',
            'negative']].agg(
                {'Name': 'first', 
                'Top_n_words': 'first',
                'review_id': 'nunique',
                'positive': 'sum',
                'neutral': 'sum',
                'negative': 'sum'}).rename(columns=
                                            {'review_id': 'num_reviews'})
        
        #final dataframe... 1 row per topic
        #topic, name, num_reviews, positive, neutral, negative,
        #dont show outliers (Topic == -1)

        #note fix if -1 index (no outliers)
        return final_df
    

    # def get_representative_docs(self, documents: Chunks, topic_num)->pd.DataFrame:
    #     texts = df['chunks'].tolist()
    #     review_ids = df['review_ids'].tolist()
    #     df = self.topic_model.get_document_info(texts)
    #     df['review_ids'] = review_ids

    #     return df[(df['Topic']==topic_num)  & df['Representative_document']][['Document', 'review_id']]
