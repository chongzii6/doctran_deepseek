from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter

file_path = u'D:\\dev\\AI\\files\\演示文件\\会议纪要\\中国地质科学院探矿工艺研究所安委会会议纪要(20240929).pdf'
# loader = PyPDFLoader(file_path, extract_images = True,)
# pages = []
# for page in loader.load():
#     pages.append(page)

loader = PyMuPDFLoader(
    file_path,
    mode="single",
    extract_images = True,
)
docs = loader.load()

print(f"{docs[0].metadata}\n")
print(docs[0].page_content)

# text_splitter = SpacyTextSplitter(pipeline='zh_core_web_sm', separator="|")
# sp_docs = text_splitter.split_documents(docs)
# print(sp_docs[0].page_content)

# import spacy
# from spacy.tokens import Span, Token 
# # 加载Spacy的模型
# nlp = spacy.load('zh_core_web_sm')
# # 对文本进行实体识别
# doc = nlp(docs[0].page_content)

# merged_entities = []
# current_entity = None

# for token in doc:
#     if token.ent_type_ != "":
#         if current_entity is None:
#             current_entity = token
#         else:
#             et = None
#             if isinstance(current_entity, Token):
#                 et = current_entity
#             elif isinstance(current_entity, Span):
#                 et = current_entity[0]
            
#             if token.ent_type_ == et.ent_type_:
#                 current_entity = doc[et.i:token.i+1]
#             else:
#                 merged_entities.append(current_entity)
#                 current_entity = token

#         # elif token.ent_type_ == current_entity.ent_type_:
#         #     current_entity = doc[current_entity.i:token.i+1]
#         # else:
#         #     merged_entities.append(current_entity)
#         #     current_entity = token
#     else:
#         if current_entity is not None:
#             merged_entities.append(current_entity)
#             current_entity = None

# # 处理最后一个实体
# if current_entity is not None:
#     merged_entities.append(current_entity)

# for entity in merged_entities:
#     print(entity.text, entity.label_)

# from pdf2docx import Converter

# docx_file = u'D:\\dev\\AI\\files\\演示文件\\会议纪要\\中国地质科学院探矿工艺研究所安委会会议纪要(20240929).docx'

# 创建Converter对象并进行转换
# cv = Converter(file_path)
# cv.convert(docx_file, start=0, end=None)
# cv.close()

# from langchain_community.document_loaders import Docx2txtLoader

# loader = Docx2txtLoader(docx_file)
# data = loader.load()
# print(data[0].metadata)
# print(data[0].page_content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from pydantic import BaseModel, Field
from typing import List
import os

class Classification(BaseModel):
    names: List[str] = Field(description=u"中文名字")
    title: str = Field(description=u"文档标题")
    subject: str = Field(description=u"会议主题")
    time: str = Field(description=u"会议时间")
    publisher: str = Field(description=u"文档发布单位")
    chairperson: List[str] = Field(description=u"主持人")
    attendees: List[str] = Field(description=u"参会人")
    recorder: List[str] = Field(description=u"记录人")
    cc: List[str] = Field(description=u"抄送")


dashscope_api_key = os.getenv("DASHSCOPE_KEY")
agent_model = "qwen-max-latest"

extraction_prompt = ChatPromptTemplate.from_template(
"""
分析以下文本，提取包含的人名，文档标题，会议主题，文档发布单位，主持人，参会人，记录人，抄送。
<文本>
{input}
</文本>
"""
)

llm = ChatTongyi(temperature=0.6, model=agent_model, api_key=dashscope_api_key, streaming=True).with_structured_output(
    Classification
)

prompt = extraction_prompt.invoke(docs[0].page_content)
output = llm.invoke(prompt)
print(output, type(output))
