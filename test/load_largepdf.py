from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader

file_path = u'D:\\dev\\AI\\files\\演示文件\\政府工作报告(20240312).pdf'
# loader = PyPDFLoader(file_path)
# pages = []
# for page in loader.load():
#     pages.append(page)

loader = PyMuPDFLoader(
    file_path,
    # mode="single",
    extract_images = True,
)
docs = loader.load()

print(f"{docs[0].metadata}\n")
print(docs[0].page_content)
