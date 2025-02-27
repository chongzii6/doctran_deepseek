# Doctran 中文修订版

<p align="center">
  <p align="center"><b>Doc</b>ument <b>tran</b>sformation 框架 - 使用 LLM 通过自然语言指令处理复杂文本</p>
</p>

<p align="center">
<a href="https://github.com/psychic-api/doctran/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/static/v1?label=license&message=MIT&color=blue" alt="License">
</a>
<a href="https://github.com/psychic-api/doctran/issues" target="_blank">
    <img src="https://img.shields.io/github/issues/psychic-api/doctran?color=blue" alt="Issues">
</a>
</p>

## 修订说明

**重要提示：这是 Doctran 的中文修订版，由于原项目作者已不再积极维护，我们提供了以下关键修复：**

1. **适配 OpenAI 1.0+ API SDK**：更新了代码以兼容 OpenAI 的最新 API 架构
2. **支持 DeepSeek 模型**：增加了对 DeepSeek 模型的支持，特别是 DeepSeek V2.5
3. **兼容 Pydantic V1/V2**：修改代码以同时支持 Pydantic 的 V1 和 V2 版本
4. **修复 tiktoken 兼容性问题**：增加了对非标准模型的 token 计算支持
5. **增强 JSON 解析能力**：改进了对不规范 JSON 输出的处理

**注意事项：**
- 本修订版尚未上传至 PyPI，请通过 GitHub 仓库安装
- DeepSeek R1 和 V3 不支持 tool calling 功能，请仅使用 DeepSeek V2.5 模型
- 详细使用方法请参考 `Examples.ipynb` 文件

## 简介

有些应用场景需要解析文档，而这些场景中人类级别的判断比处理速度更重要。例如标记交易或从文本中提取语义信息。在这些情况下，正则表达式可能过于僵化，而 LLM 则是理想选择。

Doctran 可以看作是一个由 LLM 驱动的黑盒，将混乱的字符串输入转换为整洁、干净、标记良好的字符串输出。另一种理解方式是，它是 OpenAI 函数调用功能的模块化、声明式包装器，大大改善了开发者体验。

## 示例

克隆或下载 [`examples.ipynb`](/examples.ipynb) 获取交互式演示。

#### Doctran 将混乱、非结构化的文本

```
<doc type="Confidential Document - For Internal Use Only">
<metadata>
<date> &#x004A; &#x0075; &#x006C; &#x0079; &#x0020; &#x0031; , &#x0032; &#x0030; &#x0032; &#x0033; </date>
<subject> Updates and Discussions on Various Topics; </subject>
</metadata>
<body>
<p>Dear Team,</p>
<p>I hope this email finds you well. In this document, I would like to provide you with some important updates and discuss various topics that require our attention. Please treat the information contained herein as highly confidential.</p>
...
```

#### 转换为半结构化文档，优化向量搜索

```json
{
  "topics": ["Security and Privacy", "HR Updates", "Marketing", "R&D"],
  "summary": "The document discusses updates on security measures, HR, marketing initiatives, and R&D projects. It commends John Doe for enhancing network security, welcomes new team members, and recognizes Jane Smith for her customer service. It also mentions the open enrollment period for employee benefits, thanks Sarah Thompson for her social media efforts, and announces a product launch event on July 15th. David Rodriguez is acknowledged for his contributions to R&D. The document emphasizes the importance of confidentiality.",
  "contact_info": [
    {
      "name": "John Doe",
      "contact_info": {
        "phone": "",
        "email": "john.doe@example.com"
      }
    },
    ...
  ],
  "questions_and_answers": [
    {
      "question": "What is the purpose of this document?",
      "answer": "The purpose of this document is to provide important updates and discuss various topics that require the team's attention."
    },
    ...
  ]
}
```

## 开始使用

由于本修订版尚未发布到 PyPI，请从 GitHub 仓库安装：

```bash
git clone https://github.com/your-username/doctran-cn.git
cd doctran-cn
pip install -e .
```

基本用法：

```python
from doctran import Doctran

# 使用 OpenAI 模型
doctran = Doctran(
    openai_api_key="your_openai_api_key",
    openai_model="gpt-3.5-turbo"  # 或 "gpt-4"
)

# 使用 DeepSeek 模型 (仅支持 V2.5)
doctran = Doctran(
    openai_api_key="your_deepseek_api_key",
    openai_model="deepseek-ai/DeepSeek-V2.5",
    openai_base_url="https://api.deepseek.com/v1"  # DeepSeek API 端点
)

# 解析文档
document = doctran.parse(content="你的文本内容")
```

## 链式转换

Doctran 设计为易于链式调用文档转换。例如，您可能希望先从文档中删除所有 PII（个人身份信息），然后再将其发送到 OpenAI 进行摘要处理。

链式调用中的顺序很重要 - 先调用的转换将先执行，其结果将传递给下一个转换。

```python
document = await document.redact(entities=["EMAIL_ADDRESS", "PHONE_NUMBER"]).extract(properties).summarize().execute()
```

## 文档转换器

### Extract（提取）

根据任何有效的 JSON schema，使用 LLM 函数调用从文档中提取结构化数据。

```python
from doctran import ExtractProperty

properties = ExtractProperty(
    name="millenial_or_boomer", 
    description="预测此文档是由千禧一代还是婴儿潮一代撰写的",
    type="string",
    enum=["millenial", "boomer"],
    required=True
)
document = await document.extract(properties=properties).execute()
```

### Redact（编辑）

使用 spaCy 模型从文档中删除姓名、电子邮件、电话号码和其他敏感信息。在本地运行，避免将敏感数据发送到第三方 API。

```python
document = await document.redact(entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"]).execute()
```

### Summarize（摘要）

总结文档中的信息。可以通过 `token_limit` 配置摘要的大小，但 LLM 可能不会严格遵守。

```python
document = await document.summarize().execute()
```

### Refine（精炼）

从文档中删除所有信息，除非它与特定主题集相关。

```python
document = await document.refine(topics=['营销', '会议']).execute()
```

### Translate（翻译）

将文本翻译成另一种语言。

```python
document = await document.translate(language="中文").execute()
```

### Interrogate（问答转换）

将文档中的信息转换为问答格式。最终用户查询通常以问题形式出现，因此将信息转换为问题并从这些问题创建索引，在使用向量数据库进行上下文检索时通常会产生更好的结果。

```python
document = await document.interrogate().execute()
```

## DeepSeek 模型使用注意事项

在使用 DeepSeek 模型时，请注意以下几点：

1. 仅支持 DeepSeek V2.5 模型，因为 DeepSeek R1 和 V3 不支持 tool calling 功能
2. 建议增加 token 限制，以避免输出被截断：

```python
doctran = Doctran(
    openai_api_key="your_deepseek_api_key",
    openai_model="deepseek-ai/DeepSeek-V2.5",
    openai_base_url="https://api.deepseek.com/v1",
    openai_token_limit=16000  # 增加 token 限制
)
```

3. 如果遇到 JSON 解析错误，可以尝试调整温度参数：

```python
# 在 DoctranConfig 中设置
config = DoctranConfig(
    openai_api_key="your_deepseek_api_key",
    openai_model="deepseek-ai/DeepSeek-V2.5",
    openai_base_url="https://api.deepseek.com/v1",
    openai_temperature=0  # 设置为 0 以获得更确定性的输出
)
```

## 贡献

欢迎对 Doctran 中文修订版做出贡献！最好的开始方式是贡献新的文档转换器。不依赖 API 调用（例如 OpenAI）的转换器特别有价值，因为它们可以快速运行且不需要任何外部依赖。

### 添加新的文档转换器

贡献新转换器非常简单：

1. 添加一个继承 `DocumentTransformer` 或 `OpenAIDocumentTransformer` 的新类
2. 实现 `__init__` 和 `transform` 方法
3. 在 `DocumentTransformationBuilder` 和 `Document` 类中添加相应方法以启用链式调用