# azure_search_tool.py
from azure.search.documents import SearchClient

class AzureSearchTool:
    def __init__(self, service_endpoint: str, admin_key: str, index_name: str, model=None):
        """Initialize with Azure search credentials and optional embedding model."""
        self.index_name = index_name
        self.service_endpoint = service_endpoint
        self.search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(admin_key))
        # Use the same model as in indexing for embedding queries
        if model:
            self.model = model
        else:
            # Default to the same MiniLM model if none provided
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve top_k most similar documents for the query from Azure Search.
        Returns a list of results with id, content, etc.
        """
        # Generate embedding for the query using the same model
        query_embedding = self.model.encode(query).tolist()
        # Perform vector search on Azure index
        results = self.search_client.search(
            vector_queries=[{
                "kind": "vector",
                "value": query_embedding,
                "fields": ["embedding"],
                "k": top_k
            }],
            select=["id", "file", "content"]
        )
        # Collect results into a list of dictionaries
        retrieved = []
        for result in results:
            retrieved.append({
                "id": result["id"],
                "file": result.get("file"),
                "content": result["content"]
            })
        return retrieved
