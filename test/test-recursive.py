
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

pdf_path = u"D:\\dev\\AI\\files\\关于对海淀“十四五和二零三五年远景目标”建言献策、征集意见的结果分析报告.pdf"

loader = PyMuPDFLoader(
    pdf_path,
    mode="single",
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, length_function=len
)

texts = text_splitter.split_documents(docs)
print("--- splitted docs=", len(texts))

i = 0
for i in range(len(texts)):
    print(f"\n----------- part: {i} -----------")
    print(texts[i].page_content)
