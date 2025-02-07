# test_search.py
from azure_search_tool import AzureSearchTool

# Initialize the search tool (use the same endpoint, key, and index as before)
service_endpoint = "https://<YOUR-SEARCH-SERVICE>.search.windows.net"
admin_key = "<YOUR-ADMIN-KEY>"
index_name = "pdf-index"

retriever = AzureSearchTool(service_endpoint, admin_key, index_name)
query = "Your query text here"
results = retriever.search(query, top_k=3)

print(f"Query: {query}")
for i, res in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(f"Document ID: {res['id']}")
    print(f"Source File: {res.get('file')}")
    print(f"Content Excerpt: {res['content'][:200]}...")  # print first 200 chars
