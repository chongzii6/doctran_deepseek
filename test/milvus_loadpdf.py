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
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient, DataType, Function, FunctionType
import os.path

dashscope_api_key = os.getenv("DASHSCOPE_KEY")
agent_model = os.getenv("AGENT_MODEL", default="qwen-max-latest")
generating_model = os.getenv("GENERATING_MODEL", default="qwen2.5-72b-instruct")

ollama_url = os.getenv("OLLAMA_URL", default="http://192.168.8.2:11434")
milvus_url = os.getenv("MILVUS_URL", default="http://192.168.8.2:19530")
collection_name = os.getenv("MILVUS_COLLECTION", default="milvus_meeting_minutes")
base_url = os.getenv("OPENAI_BASEURL", default="https://dashscope.aliyuncs.com/compatible-mode/v1")



# class Classification(BaseModel):
#     names: List[str] = Field(description=u"中文名字")
#     title: str = Field(description=u"文档标题")
#     subject: str = Field(description=u"会议主题")
#     time: str = Field(description=u"会议时间")
#     publisher: str = Field(description=u"文档发布单位")
#     chairperson: List[str] = Field(description=u"主持人")
#     attendees: List[str] = Field(description=u"参会人")
#     recorder: List[str] = Field(description=u"记录人")
#     cc: List[str] = Field(description=u"抄送")

# extraction_prompt = ChatPromptTemplate.from_template(
# """
# 分析以下文本，提取包含的人名，文档标题，会议主题，文档发布单位，主持人，参会人，记录人，抄送。
# <文本>
# {input}
# </文本>
# """
# )
class Classification(BaseModel):
    names: List[str] = Field(description=u"名字")
    subject: str = Field(description=u"会议主题")
    time: str = Field(description=u"会议时间")
    publisher: str = Field(description=u"文档发布单位")
    cc: List[str] = Field(description=u"抄送")

extraction_prompt = ChatPromptTemplate.from_template(
"""
分析以下文本，提取包含的所有名字，会议主题，会议时间，文档发布单位，抄送。
<文本>
{input}
</文本>
"""
)

def extract_props_openai(openai, func_schema, model, text):
    prompt = extraction_prompt.invoke({"input": text})
    content = prompt.to_messages()[0].content

    # schema = Classification.model_json_schema()
    # func_schema = {
    #     "name": "Classification",
    #     "parameters": schema,
    # }

    # openai = OpenAI(base_url=base_url, api_key=dashscope_api_key)
    response = openai.chat.completions.create(temperature=0, model=model, 
                messages=[{"role": "user", "content": content}],
                tools=[{
                    "type": "function",
                    "function": func_schema,
                    }
                ],
    )

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
        if arguments_dict:
            return arguments_dict
            # arguments_dict = {"error": "模型返回了空的 JSON 结构"}

        # print(tool_call)
        # print(arguments_dict)

    except Exception as e:
        raise Exception("无法解析模型返回的工具调用: " + str(e))    

def load_pdfs(dir):
    loader = PyPDFDirectoryLoader(
        path = dir,
        glob = "*.pdf",
        silent_errors = False,
        load_hidden = False,
        recursive = False,
        extract_images = False,
        password = None,
        mode = "single",
        # images_to_text = None,
        headers = None,
        extraction_mode = "plain",
        # extraction_kwargs = None,
    )

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # 使用LangChain将输入文档安照chunk_size切分
    all_splits = text_splitter.split_documents(docs)
    return all_splits

    # text_contents = [doc.page_content for doc in all_splits]
    # metadatas = [{"source": os.path.basename(doc.metadata["source"])} for doc in all_splits]

def transform_docs(docs):
    schema = Classification.model_json_schema()
    func_schema = {
        "name": "Classification",
        "parameters": schema,
    }

    openai_LLM = OpenAI(base_url=base_url, api_key=dashscope_api_key)
    text_contents = []
    metadatas = []

    for doc in docs:
        text = doc.page_content
        source = os.path.basename(doc.metadata["source"])
        props = extract_props_openai(openai_LLM, func_schema, agent_model, text)
        if not props:
            break

        props['source'] = source
        text_contents.append(text)
        metadatas.append(props)

    return text_contents, metadatas

def embedding_text(text_contents):
    # dense_dim = 1024  # for bge-m3
    # embedding_model = "bge-m3"
    dense_dim = 768  # for nomic-embed-text
    embedding_model = "nomic-embed-text"
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_url)
    vectors = embeddings.embed_documents(text_contents)
    return vectors, dense_dim

def upload_milvus(text_contents, metadatas):
    client = MilvusClient(
        uri=milvus_url,
    )

    schema = MilvusClient.create_schema(
        enable_dynamic_field=True,
    )

    analyzer_params = {
        "type": "chinese"
    }

    vectors, dense_dim = embedding_text(text_contents)

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
    # Insert data
    res = client.insert(
        collection_name=collection_name,
        data=data
    )

    print(f"生成 {len(vectors)} 个向量，维度：{len(vectors[0])}")

def _showhelp():
    print('''%s [options] <folder_path>\nOption: \n\t-h --help: show help \n\t--key=<key>: openai access key
\t--model=<model_id>: agent model to use to extract properties
\t--ollama_url=<url>: ollama url for embedding
\t--milvus_url=<url>: milvus url
\t--openai_url=<url>: openai api compatible url
\t--collection=<name>: milvus collection name\n
folder_path: upload files in folder\n''' % os.path.basename(argv[0]))

if __name__ == "__main__":
    path = u"D:\\dev\\AI\\files\\演示文件\\会议纪要"

    from sys import argv
    import getopt

    opts, args = getopt.getopt(argv[1:],'-h',['help','key=','model=','ollama_url=','milvus_url=','openai_url=','collection='])
    for opt_name, opt_value in opts:
        if opt_name in ('-h','--help'):
            _showhelp()
            exit()
        if opt_name == '--key':
            dashscope_api_key = opt_value
        elif opt_name == '--model': 
            agent_model = opt_value
        elif opt_name == '--ollama_url': 
            ollama_url = opt_value
        elif opt_name == '--milvus_url': 
            milvus_url = opt_value
        elif opt_name == '--openai_url': 
            base_url = opt_value
        elif opt_name == '--collection': 
            collection_name = opt_value
    
    if len(args) > 0:
        path = args
        all_splits = load_pdfs(path)
        text_contents, metadatas = transform_docs(all_splits)
        upload_milvus(text_contents, metadatas)
    else:
        _showhelp()

