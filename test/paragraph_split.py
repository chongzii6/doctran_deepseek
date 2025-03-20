from langchain.text_splitter import NLTKTextSplitter
from langchain.text_splitter import SpacyTextSplitter

# 你的文本
text = u"新兴产业和新兴业态是竞争高地。要实施高端装备、信息网络、集成电路、新能源、新材料、生物医药、航空发动机、燃气轮机等重大项目，把一批新兴产业培育成主导产业。制定“互联网+”行动计划，推动移动互联网、云计算、大数据、物联网等与现代制造业结合，促进电子商务、工业互联网和互联网金融健康发展，引导互联网企业拓展国际市场。国家已设立400亿元新兴产业创业投资引导基金，要整合筹措更多资金，为产业创新加油助力。"
# text_splitter = NLTKTextSplitter(chunk_size=100, chunk_overlap=20)
text_splitter = SpacyTextSplitter(pipeline='zh_core_web_sm', separator="|")
docs = text_splitter.split_text(text)
texts = []

for d in docs:
    texts.extend(d.split("|"))
    # texts.extend(d.page_content.split("|"))

print(len(texts))
print(texts)
