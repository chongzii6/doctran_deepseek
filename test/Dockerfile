FROM python:3.12.9-slim

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
    pip config --user set global.progress_bar off

RUN pip install --no-cache-dir langchain-community pypdf pymilvus tqdm pillow
RUN pip install --no-cache-dir langchain-docling openpyxl spacy ollama langchain-milvus
RUN pip install --no-cache-dir dashscope matplotlib langchain-ollama langchain-text-splitters
RUN pip install --no-cache-dir httpx huggingface-hub
RUN pip install -U pydantic

COPY ./zh_core_web_sm-3.8.0.tar.gz /tmp
RUN ls -al /tmp && pip install /tmp/zh_core_web_sm-3.8.0.tar.gz
RUN mkdir /app
COPY ./combined_query.py /app

WORKDIR /app
ENV OPENBLAS_NUM_THREADS=1
RUN pip install langchain-community==0.3.19
