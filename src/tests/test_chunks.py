from sentopic.chunks import Chunk, Chunks, ChunkAPI
import pytest
import numpy as np 

@pytest.fixture
def chunks():
    api = ChunkAPI('http://127.0.0.1:8000/')
    chunks = api.get_chunks(offering_id = 1201662)
    return chunks

def test_get_chunks(chunks):
    assert isinstance(chunks, Chunks) 
    for chunk in chunks:
        assert isinstance(chunk.embedding, np.ndarray)

def test_get_embeddings(chunks):
    embeddings = chunks.get_embeddings()

    assert isinstance(embeddings, np.ndarray)
    #array is float 64, 384 values
    for embedding in embeddings:
        assert embedding.dtype =='float64'
        assert embedding.shape[0] == 384


def test_get_texts(chunks):
    texts = chunks.get_texts()
    assert isinstance(texts, list)
    for text in texts:
        assert isinstance(text, str)
