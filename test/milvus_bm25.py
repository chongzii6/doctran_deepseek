from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient, DataType, Function, FunctionType
import os.path

dashscope_api_key = "sk-fe19314cfe7d424d9c8ec7c52f3dcd88"
milvus_url = "192.168.8.2"
user_name = "root"
password = "<YOUR_PASSWORD>"
collection_name = "milvus_bm25"

# loader = WebBaseLoader([
#     'https://raw.githubusercontent.com/milvus-io/milvus-docs/refs/heads/v2.5.x/site/en/about/overview.md'
# ])

text_loader_kwargs = {"autodetect_encoding": True}
path = u"D:\\dev\\AI\\files\\演示文件\\工作报告（2015-2025）"
loader = DirectoryLoader(path, glob="**/*.txt", show_progress=True, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)

# 使用LangChain将输入文档安照chunk_size切分
all_splits = text_splitter.split_documents(docs)

# embeddings = DashScopeEmbeddings(
#     model="text-embedding-v2", dashscope_api_key=dashscope_api_key
# )
# dense_dim = 1536
dense_dim = 768
# embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://192.168.8.2:11434")

dense_dim = 1024
embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://192.168.8.2:11434")

text_contents = [doc.page_content for doc in all_splits]
metadatas = [{"source": os.path.basename(doc.metadata["source"])} for doc in all_splits]

vectors = embeddings.embed_documents(text_contents)


# client = MilvusClient(
#     uri=f"http://{milvus_url}:19530",
#     token=f"{user_name}:{password}",
# )
client = MilvusClient(
    uri=f"http://{milvus_url}:19530",
)

schema = MilvusClient.create_schema(
    enable_dynamic_field=True,
)

analyzer_params = {
    "type": "chinese"
}

# Add fields to schema
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True, analyzer_params=analyzer_params, enable_match=True)
schema.add_field(field_name="sparse_bm25", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)
schema.add_field(field_name="metadata", datatype=DataType.JSON)

bm25_function = Function(
   name="bm25",
   function_type=FunctionType.BM25,
   input_field_names=["text"],
   output_field_names="sparse_bm25",
)
schema.add_function(bm25_function)

index_params = client.prepare_index_params()

# Add indexes
index_params.add_index(
    field_name="dense",
    index_name="dense_index",
    index_type="IVF_FLAT",
    metric_type="IP",
    params={"nlist": 128},
)

index_params.add_index(
    field_name="sparse_bm25",
    index_name="sparse_bm25_index",
    index_type="SPARSE_WAND",
    metric_type="BM25"
)

if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

# Create collection
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params
)

data = [
    {"dense": vectors[idx], "text": doc, "metadata": metadatas[idx]}
    for idx, doc in enumerate(text_contents)
]

print(data)

# Insert data
res = client.insert(
    collection_name=collection_name,
    data=data
)

print(f"生成 {len(vectors)} 个向量，维度：{len(vectors[0])}")
