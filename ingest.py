pip install azure-search-documents PyMuPDF sentence-transformers

# pdf_utils.py
import fitz  # PyMuPDF

def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    # Group words into chunks of desired size
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


# embedding_utils.py
from sentence_transformers import SentenceTransformer

# Load the model (this downloads the model if not already available)
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text: str) -> list[float]:
    """Generate embedding vector for a given text using the SentenceTransformer model."""
    embedding = model.encode(text)  # NumPy array
    # Convert to Python list for JSON serialization (Azure SDK expects list of floats)
    return embedding.tolist()

# azure_index.py
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchField, SearchFieldDataType
from azure.search.documents.indexes.models import VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile

# Configure your service and index names
SERVICE_ENDPOINT = "https://<YOUR-SEARCH-SERVICE>.search.windows.net"
ADMIN_KEY = "<YOUR-ADMIN-KEY>"
INDEX_NAME = "pdf-index"

def create_vector_index(service_endpoint: str, admin_key: str, index_name: str, vector_dim: int):
    """Create an Azure Search index with a vector field for embeddings."""
    credential = AzureKeyCredential(admin_key)
    index_client = SearchIndexClient(service_endpoint, credential)
    
    # Define index fields: an ID, content text, and the embedding vector
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),           # document ID (key)
        SimpleField(name="file", type=SearchFieldDataType.String, filterable=True),  # filename or document identifier
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),  # full text (searchable for fallback keyword search)
        SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
                    searchable=True, vector_search_dimensions=vector_dim, 
                    vector_search_profile_name="my-vector-profile")                # vector field for embeddings
    ]
    # Configure HNSW algorithm for vector search
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="my-vector-algo", kind="hnsw", 
                                               parameters={"m": 4, "efConstruction": 200, "efSearch": 100, "metric": "cosine"})],
        profiles=[VectorSearchProfile(name="my-vector-profile", algorithm_configuration_name="my-vector-algo")]
    )
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    # Create or update the index on Azure
    index_client.create_or_update_index(index)
    print(f"Index '{index_name}' created or updated successfully.")


# ingest_pdfs.py
import os, uuid
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from pdf_utils import extract_text_from_pdf, chunk_text
from embedding_utils import generate_embedding
from azure_index import create_vector_index, SERVICE_ENDPOINT, ADMIN_KEY, INDEX_NAME

def ingest_pdfs(directory: str):
    """Process all PDF files in the given directory and upload to Azure Search index."""
    # Ensure index exists
    create_vector_index(SERVICE_ENDPOINT, ADMIN_KEY, INDEX_NAME, vector_dim=384)
    # Initialize search client for uploading documents
    search_client = SearchClient(SERVICE_ENDPOINT, INDEX_NAME, AzureKeyCredential(ADMIN_KEY))
    
    docs_to_upload = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(file_path)
            chunks = chunk_text(text, chunk_size=500)
            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue  # skip empty chunks
                doc_id = f"{filename}-{idx}"  # unique ID per file chunk
                vector = generate_embedding(chunk)
                document = {
                    "id": doc_id,
                    "file": filename,
                    "content": chunk,
                    "embedding": vector  # list of floats
                }
                docs_to_upload.append(document)
    
    # Upload in batches to avoid too large payloads
    batch_size = 1000  # adjust as needed
    for i in range(0, len(docs_to_upload), batch_size):
        batch = docs_to_upload[i : i + batch_size]
        result = search_client.upload_documents(documents=batch)
        if result.errors:
            for error in result.errors:
                print(f"Failed to upload document: {error}")
    print(f"Uploaded {len(docs_to_upload)} documents (chunks) to index '{INDEX_NAME}'.")
    
# Example usage:
if __name__ == "__main__":
    ingest_pdfs("path/to/pdf_folder")
