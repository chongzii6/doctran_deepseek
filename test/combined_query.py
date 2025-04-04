from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from pydantic import BaseModel, Field
from typing import List
from pymilvus import MilvusClient, DataType, Function, FunctionType
from langchain_core.documents import Document
from langchain.text_splitter import SpacyTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os, json
from langchain_openai import ChatOpenAI
from openai import OpenAI

dashscope_api_key = os.getenv("DASHSCOPE_KEY")
agent_model = os.getenv("AGENT_MODEL", default="qwen-max-latest")
generating_model = os.getenv("GENERATING_MODEL", default="qwen2.5-72b-instruct")

milvus_url = os.getenv("MILVUS_URL", default="192.168.8.2")
collection_name = os.getenv("MILVUS_COLLECTION", default="milvus_bm25")
base_url = os.getenv("OPENAI_BASEURL", default="https://dashscope.aliyuncs.com/compatible-mode/v1")

#
extraction_prompt = ChatPromptTemplate.from_template(
"""
分析以下问题，提取问题的关键字，如果有多个，按询问的意图高低排序。
<问题>
{input}
</问题>
"""
)

class Classification(BaseModel):
    keyword: List[str] = Field(description=u"查询的关键字")

def extra_keyword_openai(query) -> List[str]:
    prompt = extraction_prompt.invoke({"input": query})
    content = prompt.to_messages()[0].content
    # print(content)

    schema = Classification.model_json_schema()
    func_schema = {
        "name": "Classification",
        "parameters": schema,
    }

    openai = OpenAI(base_url=base_url, api_key=dashscope_api_key)

    # print(json.dumps(func_schema, indent=2))
    response = openai.chat.completions.create(temperature=0, model=agent_model, 
                messages=[{"role": "user", "content": content}],
                tools=[{
                    "type": "function",
                    "function": func_schema,
                    }
                ],
                # tool_choice={"type": "function", "function": {"name": schema["name"]}}
    )

    # print(response)

    # 解析工具调用结果
    try:
        # 确保正确访问工具调用结果
        tool_call = response.choices[0].message.tool_calls[0]
        arguments = tool_call.function.arguments

        try:
            arguments_dict = json.loads(arguments)
        except json.JSONDecodeError:
            print("尝试修复 JSON 格式问题...")

        # 如果 arguments_dict 为空，创建默认结构
        if not arguments_dict:
            arguments_dict = {"error": "模型返回了空的 JSON 结构"}

        # print(tool_call)
        # print(arguments_dict)
        return arguments_dict['keyword']

    except Exception as e:
        raise Exception("无法解析模型返回的工具调用: " + str(e))    

def extra_keyword_tongyi(query) -> List[str]:
    # LLM\n",
    llm = ChatTongyi(temperature=0.6, model=agent_model, api_key=dashscope_api_key, streaming=True).with_structured_output(
        Classification
    )
    output = llm.invoke(user_query)
    print(output)

    if len(output.keyword) == 0:
        print("无法识别查询关键字, 失败!")
        return
    
    return output.keyword

def rag_by_keyword(user_query):
    print("问题: ", user_query)

    # kws = extra_keyword_tongyi(user_query)
    kws = extra_keyword_openai(user_query)
    print(f'keywords={kws}')

    if len(kws) == 0:
        print("无法识别查询关键字, 失败!")
        return

###
    mclient = MilvusClient(
        uri=f"http://{milvus_url}:19530",
    )

    keyword = kws[0]
    filter = f"text like '%{keyword}%'"
    # print(f'filter={filter}')

    matching_documents = []
    hybrid_search_res = mclient.query(
        collection_name=collection_name,
        filter=filter,
        output_fields=["text", "metadata"]
    )

    text_splitter = SpacyTextSplitter(pipeline='zh_core_web_sm', separator="|")

    for hits in hybrid_search_res:  # Use the correct variable here
        t = hits['text']
        docs = text_splitter.split_text(t)

        texts = []
        for d in docs:
            texts.extend(d.split("|"))

        for txt in texts:
            if keyword in txt:
                matching_documents.append(Document(page_content=txt, metadata=hits['metadata']))  # Extract text content to the context list

    ll = len(matching_documents)
    print(f"got {ll} matched records")
    if ll == 0:
        print("无法找到与关键字有关联的记录, 失败!")
        return

    sources = []
    for i, md in enumerate(matching_documents):
    #     src = f'''<source><source_id>{i}</source_id>
    # <source_context>{md.metadata['source']}</source_context></source>
    # <source><source_id>{i}</source_id>
    # <source_context>{md.page_content}</source_context></source>
    # '''
        src = f'''<source><source_id>{md.metadata['source']}</source_id>
    <source_context>{md.page_content}</source_context></source>
    '''
        sources.append(src)

    context = ''.join(sources)
    # print(context)

    system_prompt = PromptTemplate.from_template(
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

    # generatingLLM = ChatTongyi(temperature=0.6, model=generating_model, 
    #                         api_key=dashscope_api_key, streaming=True)
    generatingLLM = ChatOpenAI(temperature=1.2, model=generating_model, base_url=base_url, 
                            api_key=dashscope_api_key, streaming=True)

    # Step 8: Create Chain
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | system_prompt
        | generatingLLM
        | StrOutputParser()
    )

    response = chain.invoke({"context": context, "question":user_query})
    print(response)

# query = u"关于工业互联网历年有什么不同表述"
# user_query = u"历年来关于工业互联网有什么不同表述"
while True:
    user_query = input('请输入您的问题:（exit=退出）')
    if user_query == 'exit':
        print("再见!")
        break
    else:
        rag_by_keyword(user_query)
