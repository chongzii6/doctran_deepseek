from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient, DataType, Function, FunctionType

dashscope_api_key = "sk-fe19314cfe7d424d9c8ec7c52f3dcd88"
milvus_url = "192.168.8.2"
collection_name = "milvus_bm25"

from pymilvus import MilvusClient
from pymilvus import AnnSearchRequest, RRFRanker
from langchain_community.embeddings import DashScopeEmbeddings
from dashscope import Generation

# 创建Milvus Client。
client = MilvusClient(
    uri=f"http://{milvus_url}:19530",
)

# 初始化 Embedding 模型
# embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://192.168.8.2:11434")
embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://192.168.8.2:11434")

# Define the query
query = u"关于工业互联网历年有什么不同表述"
# query = u"工业互联网"

# Embed the query and generate the corresponding vector representation
query_embeddings = embeddings.embed_documents([query])

# Set the top K result count
top_k = 15  # Get the top 5 docs related to the query

# Define the parameters for the dense vector search
search_params_dense = {
    "metric_type": "IP",
    "params": {"nprobe": 10}
}

# Create a dense vector search request
request_dense = AnnSearchRequest([query_embeddings[0]], "dense", search_params_dense, limit=top_k)

# Define the parameters for the BM25 text search
search_params_bm25 = {
    "metric_type": "BM25"
}

# Create a BM25 text search request
request_bm25 = AnnSearchRequest([query], "sparse_bm25", search_params_bm25, limit=top_k)

# Combine the two requests
# reqs = [request_dense, request_bm25]
# reqs = [request_dense]
reqs = [request_bm25]

# Initialize the RRF ranking algorithm
ranker = RRFRanker(100)

filter = f"text like '%{query}%'"

# Perform the hybrid search
hybrid_search_res = client.hybrid_search(
    collection_name=collection_name,
    reqs=reqs,
    ranker=ranker,
    limit=top_k,
    # filter=filter,
    output_fields=["text"]
)

# hybrid_search_res = client.search(
#     collection_name=collection_name,
#     anns_field="dense",
#     data=query_embeddings,
#     filter=filter,
#     search_params={"params": {"nprobe": 10}},
#     limit=2,
#     output_fields=["text"]
# )

# search_params = {
#     'params': {'drop_ratio_search': 0.2},
# }

# hybrid_search_res = client.search(
#     collection_name=collection_name,
#     data=[query],
#     anns_field='sparse_bm25',
#     limit=top_k,
#     search_params=search_params,
#     output_fields=["text"],
# )

# hybrid_search_res = client.query(
#     collection_name=collection_name,
#     filter=filter,
#     output_fields=["text"]
# )

# Extract the context from hybrid search results
context = []
print("Top K Results:")
for hits in hybrid_search_res:  # Use the correct variable here
    for hit in hits:
        context.append(hit['entity']['text'])  # Extract text content to the context list
        print(hit['entity']['text'])  # Output each retrieved document
        print("----------------------------------")


# Define a function to get an answer based on the query and context
def getAnswer(query, context):
    prompt = f'''Please answer my question based on the content within:
    ```
    {context}
    ```
    My question is: {query}.
    '''
    # Call the generation module to get an answer
    rsp = Generation.call(model='qwq-32b', prompt=prompt)
    return rsp.output.text


# Get the answer
# answer = getAnswer(query, context)

# print(answer)


# Expected output excerpt
"""
Milvus is highly scalable due to its cloud-native and highly decoupled system architecture. This architecture allows the system to continuously expand as data grows. Additionally, Milvus supports three deployment modes that cover a wide...
"""