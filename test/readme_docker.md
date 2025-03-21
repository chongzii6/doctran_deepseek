### build image
`docker build -t rag-demo-img:v1 .`

### 命令行参数：
* DASHSCOPE_KEY     # 无默认值，相当于 OPENAI_KEY
* AGENT_MODEL       ="qwen-max-latest"
* GENERATING_MODEL  ="qwen2.5-72b-instruct"
* MILVUS_URL        ="192.168.8.2"
* MILVUS_COLLECTION ="milvus_bm25"
* OPENAI_BASEURL    ="https://dashscope.aliyuncs.com/compatible-mode/v1"

### start a pod
`docker run -ti --rm --privileged -e MILVUS_URL=172.1.8.1 rag-demo-img:v1 bash`

### execute
`python combined_query.py`

