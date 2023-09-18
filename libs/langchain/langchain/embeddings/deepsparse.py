from typing import Any

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Extra

from transformers import Pipeline

DEFAULT_MODEL_NAME = "zeroshot/oneshot-minilm"

def mean_pooling(model_output, attention_mask):
    """
    Compute mean pooling of token embeddings weighted by attention mask.

    Args:
        model_output (torch.Tensor): The model's output tensor.
        attention_mask (torch.Tensor): The attention mask tensor.

    Returns:
        torch.Tensor: Mean-pooled embeddings.
    """
    import torch
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SentenceEmbeddingPipeline(Pipeline):
    
    client: Any  #: :meta private:
    
    def __init__(self, model, tokenizer):
        """
        Initialize the SentenceEmbeddingPipeline.

        Args:
            model (Any): The model for sentence embeddings.
            tokenizer (Any): The tokenizer for preprocessing.
        """
        self.model = model
        self.tokenizer = tokenizer

    def preprocess(self, inputs):
        """
        Preprocess the input texts.

        Args:
            inputs (List[str]): List of input texts.

        Returns:
            Dict[str, torch.Tensor]: Preprocessed input tensors.
        """
        encoded_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
        return encoded_inputs

    def _forward(self, model_inputs):
        """
        Forward pass through the model.

        Args:
            model_inputs (Dict[str, torch.Tensor]): Input tensors.

        Returns:
            Dict[str, torch.Tensor]: Model outputs.
        """
        outputs = self.model(**model_inputs)
        return {"outputs": outputs, "attention_mask": model_inputs["attention_mask"]}

class DeepSparseEmbeddings(BaseModel, Embeddings):
    """DeepSparse embedding models.

    To use, you should have ``optimum.deepsparse`` package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import DeepSparseEmbeddings

            model_name = "zeroshot/oneshot-minilm"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            sparse_embeddings = DeepSparseEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Initialize the DeepSparseEmbeddings.

        Args:
            model_name (str, optional): The name of the model. Defaults to DEFAULT_MODEL_NAME.
        """
        super().__init__()
        self.model_name = model_name
        try:
            import optimum.deepsparse
        except ImportError as exc:
            raise ImportError(
                "Could not import optimum.deepsparse. "
                "Please make sure it is installed correctly. "
                "Try: pip install git+https://github.com/neuralmagic/optimum-deepsparse.git"
            ) from exc
        self.init_model()
        
    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        
    def init_model(self):
        
        from optimum.deepsparse import DeepSparseModelForFeatureExtraction
        from transformers.onnx.utils import get_preprocessor
        
        sparse_model = DeepSparseModelForFeatureExtraction.from_pretrained(self.model_name, export=False)
        tokenizer = get_preprocessor(self.model_name)
        self.model = SentenceEmbeddingPipeline(model=sparse_model, tokenizer=tokenizer)

    def embed_query(self, text):
        """
        Embed a query text.

        Args:
            text (str): The query text.

        Returns:
            List[float]: Query embeddings.
        """
        import torch.nn.functional as F
        
        encoded_text = self.model.preprocess([text])
        model_outputs = self.model._forward(encoded_text)
        sentence_embeddings = mean_pooling(model_outputs["outputs"], model_outputs['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()[0]

    def embed_documents(self, texts):
        """
        Embed a list of documents.

        Args:
            texts (List[str]): List of document texts.

        Returns:
            List[List[float]]: List of document embeddings.
        """
        return [self.embed_query(text) for text in texts]
