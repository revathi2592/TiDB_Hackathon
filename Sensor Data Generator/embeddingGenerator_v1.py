import os
from dotenv import load_dotenv
load_dotenv()
import vertexai
from vertexai.language_models import TextEmbeddingModel


# Create embedding
def create_embedding(input_text):
    vertexai.init(project=os.environ['PROJECT_ID'], location="us-central1")

    # Load model
    model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

    # Get the embeddings
    embeddings = model.get_embeddings([input_text],output_dimensionality=1536)

    # Each item in embeddings corresponds to an input text
    embedding_vector = embeddings[0].values
    return embedding_vector



