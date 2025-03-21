# build image
`docker build -t rag-demo-img:v1 .`

# start a pod
# 命令行参数：
# DASHSCOPE_KEY
# AGENT_MODEL       ="qwen-max-latest"
# GENERATING_MODEL  ="qwen2.5-72b-instruct"
# MILVUS_URL        ="192.168.8.2"
# MILVUS_COLLECTION ="milvus_bm25"
# OPENAI_BASEURL    ="https://dashscope.aliyuncs.com/compatible-mode/v1"

`docker run -ti --rm --privileged rag-demo-img:v1 bash`

# execute
`python combined_query.py`

#