from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient, DataType, Function, FunctionType
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_milvus import Milvus
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List

milvus_url = "192.168.8.2"
collection_name = "milvus_bm25"

dense_dim = 1024
embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://192.168.8.2:11434")

connection_args = { 'uri': f"http://{milvus_url}:19530" }

# Step 4: Create and Save the Database
# Create a vector store.
# vectorstore = Milvus(
#                     embedding_function=embeddings, 
#                     connection_args=connection_args,
#                     collection_name=collection_name,
#                 )

# # Step 5: Create Retriever
# # Search and retrieve information contained in the documents.
# retriever = vectorstore.as_retriever()
# 创建Milvus Client。

client = MilvusClient(
    uri=f"http://{milvus_url}:19530",
)

query = u"工业互联网"
filter = f"text like '%{query}%'"

class MyMilvusRetriever(BaseRetriever):
    client: MilvusClient = None
    collection_name: str = ""
    filter: str = ""

    def __init__(self, client: MilvusClient, collection_name: str, filter: str=None):
        super().__init__()
        self.client = client
        self.collection_name = collection_name
        self.filter = filter

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        matching_documents = []
        hybrid_search_res = self.client.query(
            collection_name=self.collection_name,
            filter=self.filter,
            output_fields=["text"]
        )
        for hits in hybrid_search_res:  # Use the correct variable here
            matching_documents.append(Document(page_content=hits['text']))  # Extract text content to the context list
            # for hit in hits:
            #     matching_documents.append(Document(page_content=hit['entity']['text']))  # Extract text content to the context list

        return matching_documents

retriever = MyMilvusRetriever(client, collection_name, filter)

# Step 6: Create Prompt
prompt = PromptTemplate.from_template(
    """### Task:
Respond to the user query using the provided context, incorporating inline citations in the format [source_id] **only when the <source_id> tag is explicitly provided** in the context.

### Guidelines:
- If you don't know the answer, clearly state that.
- If uncertain, ask the user for clarification.
- Respond in the same language as the user's query.
- If the context is unreadable or of poor quality, inform the user and provide the best possible answer.
- If the answer isn't present in the context but you possess the knowledge, explain this to the user and provide the answer using your own understanding.
- **Only include inline citations using [source_id] (e.g., [1], [2]) when a `<source_id>` tag is explicitly provided in the context.**
- Do not cite if the <source_id> tag is not provided in the context.  
- Do not use XML tags in your response.
- Ensure citations are concise and directly related to the information provided.

### Example of Citation:
If the user asks about a specific topic and the information is found in "whitepaper.pdf" with a provided <source_id>, the response should include the citation like so:  
* "According to the study, the proposed method increases efficiency by 20% [whitepaper.pdf]."
If no <source_id> is present, the response should omit the citation.

### Output:
Provide a clear and direct response to the user's query, including inline citations in the format [source_id] only when the <source_id> tag is present in the context.

<context>
{context}
</context>

<user_query>
{question}
</user_query>
"""
)

# Step 7: Load LLM
llm = ChatTongyi(
	api_key='sk-fe19314cfe7d424d9c8ec7c52f3dcd88',
	model='qwen-plus',
    streaming=True,
	temperature=0.2,
)

# Step 8: Create Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run Chain
# Input a query about the document and print the response.
question = u"关于工业互联网历年有什么不同表述"
response = chain.invoke(question)
print(response)

