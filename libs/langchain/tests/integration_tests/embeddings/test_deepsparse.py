"""Test DeepSparse embeddings."""
from langchain.embeddings.deepsparse import DeepSparseEmbeddings


def test_deepsparse_embedding_documents() -> None:
    """Test DeepSparse embeddings for documents."""
    documents = ["foo bar", "bar foo"]
    embedding = DeepSparseEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 384


def test_deepsparse_embedding_query() -> None:
    """Test DeepSparse embeddings for query."""
    document = "foo bar"
    embedding = DeepSparseEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 384