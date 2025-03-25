"""
title: pipeline-rbac
author: jun
description: test pipeline rbac
version: 0.0.1
licence: MIT
requirements: pymilvus, openai, langchain_openai, langchain_core
"""

from pydantic import BaseModel, Field
from typing import List, Union, Generator, Iterator, AsyncGenerator, Callable, Awaitable
from pymilvus import MilvusClient, DataType, Function, FunctionType
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough

class Pipeline:
    class Valves(BaseModel):
        MILVUS_COLLECTION: str = Field(default="milvus_meeting_minutes", description="Milvus Collection Name")
        MILVUS_URL: str = Field(default="http://192.168.8.2:19530", description="Milvus Url")
        OPENAI_API_KEY: str = Field(default="your-api-key")
        OPENAI_BASE_URL: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
        MODEL_ID: str = Field(default="qwen2.5-72b-instruct", description="Model Name")
        TEMPERATURE: float = Field(default=1.0, description="Generating Temperature")
        TOP_K: int = Field(default=10, description="top k of results of search")

    def __init__(self):
        self.valves = self.Valves()

    async def on_startup(self):
        # import os

        # This function is called when the server is started.
        self.milvusClient = MilvusClient(uri=self.valves.MILVUS_URL,)
        # self.openai = OpenAI(base_url=self.valves.OPENAI_BASE_URL, 
        #                      api_key=self.valves.OPENAI_API_KEY)
        self.generatingLLM = ChatOpenAI(temperature=self.valves.TEMPERATURE, 
                                   model=self.valves.MODEL_ID, 
                                   base_url=self.valves.OPENAI_BASE_URL, 
                                   api_key=self.valves.OPENAI_API_KEY, 
                                   streaming=True)
        self.system_prompt = PromptTemplate.from_template(
"""
### Task:
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


        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], 
             body: dict
             ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        if not user_message.startswith("### Task:\n"):
            username = body['user']['name']
            search_res = self._searchRBAC(user_message, username)
            print(f"username={username}\nuser_message={user_message}\nsearch_res={search_res}\n")
            sources = []
            for hits in search_res:
                for hit in hits:
                    print(f"hit={hit}")
                    entity=hit['entity']
                    # print(f"text={entity['text']}\nmetadata={entity['metadata']}\n")
                    src = f'''<source><source_id>{entity['metadata']['source']}</source_id>
<source_context>{entity['text']}</source_context></source>
'''
                    sources.append(src)

            if len(sources) > 0:
                context = ''.join(sources)
                chain = (
                    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                    | self.system_prompt
                    | self.generatingLLM
                )

                response = chain.invoke({"context": context, "question":user_message})
                return response.content


        response = self.generatingLLM.invoke(messages)
        # print(f"response={response}")
        return response.content

        # query_engine = self.index.as_query_engine(streaming=True)
        # response = query_engine.query(user_message)

        # return response.response_gen
    def _searchRBAC(self, user_message: str, username: str):
        filter = f'ARRAY_CONTAINS(metadata["names"], "{username}")'
        print(f'filter={filter}')

        # BM25 全文检索
        search_params = {
            'params': {'drop_ratio_search': 0.2},
        }

        hybrid_search_res = self.milvusClient.search(
            collection_name=self.valves.MILVUS_COLLECTION,
            data=[user_message],
            anns_field='sparse_bm25',
            limit=self.valves.TOP_K,
            search_params=search_params,
            filter=filter,
            output_fields=["text", "metadata"],
        )

        # filter only
        # hybrid_search_res = self.milvusClient.query(
        #     collection_name=self.valves.MILVUS_COLLECTION,
        #     filter=filter,
        #     output_fields=["text", "metadata"]
        # )

        return hybrid_search_res
