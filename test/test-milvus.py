from pymilvus import MilvusClient, DataType, Function, FunctionType

MILVUS_COLLECTION: str = "milvus_meeting_minutes"
MILVUS_URL: str = "http://192.168.8.2:19530"
TOP_K: int = 10

milvusClient = MilvusClient(uri=MILVUS_URL,)
username='冯志峰'
user_message='会议要求'
user_message='你好'

filter = f'ARRAY_CONTAINS(metadata["names"], "{username}")'
print(f'filter={filter}')

# BM25 全文检索
search_params = {
    'params': {'drop_ratio_search': 0.2},
}

# hybrid_search_res = milvusClient.search(
#     collection_name=MILVUS_COLLECTION,
#     data=[user_message],
#     anns_field='sparse_bm25',
#     limit=TOP_K,
#     search_params=search_params,
#     filter=filter,
#     output_fields=["text", "metadata"],
# )

hybrid_search_res = milvusClient.query(
    collection_name=MILVUS_COLLECTION,
    filter=filter,
    output_fields=["text", "metadata"],
)

print(f"hybrid_search_res={hybrid_search_res}")

# for hits in hybrid_search_res:
#     for hit in hits:
#         print(f"hit={hit}")
#         entity=hit['entity']
#         print(f"text={entity['text']}\nmetadata={entity['metadata']}\n")
for hits in hybrid_search_res:
    print(f"text={hits['text']}\nmetadata={hits['metadata']}\n")
