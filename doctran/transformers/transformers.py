from enum import Enum
import json
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
import tiktoken
from doctran import Document, DoctranConfig, ExtractProperty, RecognizerEntity

import tiktoken
from functools import wraps

# 保存原始函数
original_encoding_for_model = tiktoken.encoding_for_model

@wraps(original_encoding_for_model)
def patched_encoding_for_model(model_name):
    try:
        return original_encoding_for_model(model_name)
    except KeyError:
        print(f"警告：无法为模型 {model_name} 找到对应的分词器，使用默认分词器 cl100k_base")
        return tiktoken.get_encoding("cl100k_base")

# 替换函数
tiktoken.encoding_for_model = patched_encoding_for_model


class TooManyTokensException(Exception):
    def __init__(self, content_token_size: int, token_limit: int):
        super().__init__(f"OpenAI document transformation failed. The document is {content_token_size} tokens long, which exceeds the token limit of {token_limit}.")

class OpenAIChatCompletionCall(BaseModel):
    seed: Optional[str] = None
    model: str = "gpt-3.5-turbo-0613"
    messages: List[Dict[str, str]]
    temperature: int = 0
    max_tokens: Optional[int] = None

class OpenAIFunctionCall(OpenAIChatCompletionCall):
    tools: List[Dict[str, Any]]
    tool_choice: Union[str, Dict[str, Any]]

class DocumentTransformer(ABC):
    config: DoctranConfig

    def __init__(self, config: DoctranConfig) -> None:
        self.config = config

    @abstractmethod
    def transform(self, document: Document) -> Document:
        pass

def get_model_data(model):
    """兼容 Pydantic v1 和 v2 的模型数据提取"""
    if hasattr(model, "model_dump"):  # Pydantic v2
        return model.model_dump()
    return model.dict()  # Pydantic v1

class OpenAIDocumentTransformer(DocumentTransformer):
    function_parameters: Dict[str, Any]

    def __init__(self, config: DoctranConfig) -> None:
        super().__init__(config)
        self.function_parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def transform(self, document: Document) -> Document:
        """修改后的 transform 方法，增加对非标准模型的兼容性"""
        try:
            # 尝试获取模型对应的编码
            encoding = tiktoken.encoding_for_model(self.config.openai_model)
        except KeyError:
            # 如果模型不被识别，使用默认编码（如 cl100k_base，这是 GPT-4 使用的编码）
            print(f"警告：无法为模型 {self.config.openai_model} 找到对应的分词器，使用默认分词器 cl100k_base")
            encoding = tiktoken.get_encoding("cl100k_base")
        
        content_token_size = len(encoding.encode(document.transformed_content))
        try:
            if content_token_size > self.config.openai_token_limit:
                raise TooManyTokensException(content_token_size, self.config.openai_token_limit)
        except Exception as e:
            print(e)
            return document
        return self.executeOpenAICall(document)

    def executeOpenAICall(self, document: Document) -> Document:
        """修改后的 executeOpenAICall 方法，增加对非标准 JSON 格式的处理"""
        try:
            function_call = OpenAIFunctionCall(
                seed=self.config.openai_deployment_id,
                model=self.config.openai_model,
                messages=[{"role": "user", "content": document.transformed_content}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": self.function_name,
                        "description": self.function_description,
                        "parameters": self.function_parameters,
                    }
                }],
                tool_choice={"type": "function", "function": {"name": self.function_name}}
            )
            
            # 使用兼容性函数获取模型数据
            model_data = get_model_data(function_call)
            completion = self.config.openai.chat.completions.create(**model_data)
            
            # 解析工具调用结果
            try:
                # 确保正确访问工具调用结果
                tool_call = completion.choices[0].message.tool_calls[0]
                arguments = tool_call.function.arguments
                
                # 尝试修复 JSON 格式问题
                try:
                    # 先尝试直接解析
                    arguments_dict = json.loads(arguments)
                except json.JSONDecodeError:
                    print("尝试修复 JSON 格式问题...")
                    
                    # 1. 添加缺失的空格
                    fixed_arguments = re.sub(r'([a-z])([A-Z])', r'\1 \2', arguments)
                    
                    # 2. 修复引号问题
                    fixed_arguments = fixed_arguments.replace("'", '"')
                    
                    # 3. 确保所有 JSON 键都有双引号
                    fixed_arguments = re.sub(r'(\w+):', r'"\1":', fixed_arguments)
                    
                    # 4. 如果 JSON 不完整，尝试补全
                    if not fixed_arguments.strip().endswith("}"):
                        fixed_arguments += "}"
                    
                    print(f"修复后的 JSON: {fixed_arguments}")
                    
                    # 尝试解析修复后的 JSON
                    try:
                        arguments_dict = json.loads(fixed_arguments)
                    except json.JSONDecodeError as e:
                        # 如果仍然失败，根据转换器类型创建默认结构
                        print(f"JSON 解析仍然失败: {e}")
                        print("使用默认结构...")
                        
                        # 根据转换器类型创建默认结构
                        if hasattr(self, 'function_name'):
                            if self.function_name == "interrogate":
                                arguments_dict = {
                                    "questions_and_answers": [
                                        {
                                            "question": "解析失败，请检查模型输出",
                                            "answer": "解析失败，请检查模型输出"
                                        }
                                    ]
                                }
                            elif self.function_name == "extract_information":
                                # 为 DocumentExtractor 创建默认结构
                                arguments_dict = {}
                                for prop in getattr(self, 'properties', []):
                                    if prop.type == "string":
                                        arguments_dict[prop.name] = "解析失败，请检查模型输出"
                                    elif prop.type == "array":
                                        arguments_dict[prop.name] = []
                                    else:
                                        arguments_dict[prop.name] = None
                            elif self.function_name == "summarize":
                                arguments_dict = {"summary": "解析失败，请检查模型输出"}
                            elif self.function_name == "refine":
                                arguments_dict = {"refined_document": document.transformed_content}
                            elif self.function_name == "translate":
                                arguments_dict = {"translated_document": document.transformed_content}
                            elif self.function_name == "process_template":
                                arguments_dict = {"replacements": []}
                            else:
                                # 通用默认结构
                                arguments_dict = {"error": "无法解析模型输出"}
                        else:
                            arguments_dict = {"error": "无法解析模型输出"}
                
                # 如果 arguments_dict 为空，创建默认结构
                if not arguments_dict:
                    arguments_dict = {"error": "模型返回了空的 JSON 结构"}
            
            except Exception as e:
                raise Exception("无法解析模型返回的工具调用: " + str(e))
            
            # 处理解析后的参数
            try:
                first_value = next(iter(arguments_dict.values()), None)
                if len(arguments_dict) > 1 or not isinstance(first_value, str):
                    # 如果有多个参数或第一个参数不是字符串，视为提取的属性
                    document.extracted_properties = document.extracted_properties or arguments_dict
                else:
                    # 如果只有一个参数且是字符串，视为转换后的内容
                    document.transformed_content = first_value
                return document
            except Exception as e:
                raise Exception(f"处理模型输出时出错: {e}")
        except Exception as e:
            raise Exception(f"OpenAI 函数调用失败: {e}")



class DocumentExtractor(OpenAIDocumentTransformer):
    '''
    Use OpenAI function calling to extract structured data from the document.

    Returns:
        document: the original document, with the extracted properties added to the extracted_properties field
    '''
    properties: List[ExtractProperty]

    def __init__(self, *, config: DoctranConfig, properties: List[ExtractProperty]) -> None:
        super().__init__(config)
        self.properties = properties
        self.function_name = "extract_information"
        self.function_description = "Extract structured data from a raw text document."
        for prop in self.properties:
            self.function_parameters["properties"][prop.name] = {
                "type": prop.type,
                "description": prop.description,
                **({"items": prop.items} if prop.items else {}),
                **({"enum": prop.enum} if prop.enum else {}),
            }
            if prop.required:
                self.function_parameters["required"].append(prop.name)

class DocumentSummarizer(OpenAIDocumentTransformer):
    '''
    Use OpenAI function calling to summarize the document to under a certain token limit.

    Returns:
        document: the original document, with the summarized content added to the transformed_content field
    '''
    token_limit: int

    def __init__(self, *, config: DoctranConfig, token_limit: int) -> None:
        super().__init__(config)
        self.token_limit = token_limit
        self.function_name = "summarize"
        self.function_description = f"Summarize a document in under {self.token_limit} tokens."
        self.function_parameters["properties"]["summary"] = {
            "type": "string",
            "description": "The summary of the document.",
        }
        self.function_parameters["required"].append("summary")

class DocumentRedactor(DocumentTransformer):
    '''
    Use presidio to redact sensitive information from the document.

    Returns:
        document: the document with content redacted from document.transformed_content
    '''
    entities: List[str]
    spacy_model: str
    interactive: bool

    def __init__(self, *, config: DoctranConfig, entities: List[Union[RecognizerEntity, str]] = None, spacy_model: str = "en_core_web_md", interactive: bool = True) -> None:
        super().__init__(config)
        # TODO: support multiple NER models and sizes
        # Entities can be provided as either a string or enum, so convert to string in a all cases
        if spacy_model not in ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"]:
            raise Exception(f"Invalid spacy english language model: {spacy_model}")
        self.spacy_model = spacy_model
        self.interactive = interactive
        for i, entity in enumerate(entities):
            if entity in RecognizerEntity.__members__:
                entities[i] = RecognizerEntity[entity].value
            else:
                raise Exception(f"Invalid entity type: {entity}")
        self.entities = entities

    def transform(self, document: Document) -> Document:
        import spacy
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
        try:
            spacy.load(self.spacy_model)
        except OSError:
            from spacy.cli.download import download
            if not self.interactive:
                download(model="en_core_web_md")
            else:
                while True:
                    response = input(f"{self.spacy_model} model not found, but is required to run presidio-anonymizer. Download it now? (~40MB) (Y/n)")
                    if response.lower() in ["n", "no"]:
                        raise Exception(f"Cannot run presidio-anonymizer without {self.spacy_model} model.")
                    elif response.lower() in ["y", "yes", ""]:
                        print("Downloading...")
                        from spacy.cli.download import download
                        download(model="en_core_web_md")
                        break
                    else:
                        print("Invalid response.")
        text = document.transformed_content
        nlp_engine_provider = NlpEngineProvider(nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en",
                        "model_name": self.spacy_model
                        }]
        })
        nlp_engine = nlp_engine_provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        anonymizer = AnonymizerEngine()
        results = analyzer.analyze(text=text,
                                   entities=self.entities if self.entities else None,
                                   language='en')
        # TODO: Define customer operator to replace data types with numbered placeholders to differentiate between different PERSONs, EMAILs, etc
        anonymized_data = anonymizer.anonymize(text=text,
                                               analyzer_results=results,
                                               operators={"DEFAULT": OperatorConfig("replace")})

        # Extract just the anonymized text, discarding items metadata
        anonymized_text = anonymized_data.text
        document.transformed_content = anonymized_text
        return document

class DocumentRefiner(OpenAIDocumentTransformer):
    '''
    Use OpenAI function calling to remove irrelevant information from the document.

    Returns:
        Document: the refined content represented as a Doctran Document
    '''
    topics: List[str]

    def __init__(self, *, config: DoctranConfig, topics: List[str] = None) -> None:
        super().__init__(config)
        self.topics = topics
        self.function_name = "refine"
        if topics:
            self.function_description = f"Remove all information from a document that is not relevant to the following topics: -{' -'.join(self.topics)}"
        else:
            self.function_description = "Remove all irrelevant information from a document."
        self.function_parameters["properties"]["refined_document"] = {
            "type": "string",
            "description": "The document with irrelevant information removed.",
        }
        self.function_parameters["required"].append("refined_document")

class DocumentTranslator(OpenAIDocumentTransformer):
    '''
    Use OpenAI function calling to translate the document to another language.

    Returns:
        Document: the translated document represented as a Doctran Document
    '''
    language: str

    def __init__(self, *, config: DoctranConfig, language: str) -> None:
        super().__init__(config)
        self.function_name = "translate"
        self.function_description = f"Translate a document into {language}"
        self.function_parameters["properties"]["translated_document"] = {
            "type": "string",
            "description": f"The document translated into {language}."
        }
        self.function_parameters["required"].append("translated_document")

class DocumentInterrogator(OpenAIDocumentTransformer):
    '''
    Use OpenAI function calling to convert the document to a series of questions and answers.

    Returns:
        Document: the interrogated document represented as a Doctran Document
    '''
    def __init__(self, *, config: DoctranConfig) -> None:
        super().__init__(config)
        self.function_name = "interrogate"
        self.function_description = "Convert a text document into a series of questions and answers."
        self.function_parameters["properties"]["questions_and_answers"] = {
            "type": "array",
            "description": "The list of questions and answers.",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question.",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The answer.",
                    },
                },
                "required": ["index", "placeholder", "replaced_value"],
            },
        }
        self.function_parameters["required"].append("questions_and_answers")

class DocumentTemplateProcessor(OpenAIDocumentTransformer):
    '''
    Use OpenAI function calling to replace a given template pattern in the document with the instructions provided within each placeholder.

    For example: "Let's meet on {random day of the week}" -> "Let's meet on Monday"

    Returns:
        Document: the interrogated document represented as a Doctran Document
    '''
    def __init__(self, *, config: DoctranConfig, template_regex: str) -> None:
        super().__init__(config)
        try:
            re.compile(template_regex)
        except re.error as e:
            raise Exception(f"Provided template pattern is not valid regex: {e}")

        self.template_regex = template_regex
        self.function_name = "process_template"
        self.function_description = f"Find and replace placeholders that match the regex '{template_regex}' with the instructions provided within each placeholder."
        self.function_parameters["properties"]["replacements"] = {
            "type": "array",
            "description": "A list of replacements that should occur in the template",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The index of the replacement, in the order each placeholder appears in the template.",
                    },
                    "placeholder": {
                        "type": "string",
                        "description": "The original placeholder with instructions on how to replace it.",
                    },
                    "replaced_value": {
                        "type": "string",
                        "description": "The value to replace the placeholder with, based on the instructions provided within the placeholder.",
                    },
                },
                "required": ["question", "answer"],
            },
        }
        self.function_parameters["required"].append("replacements")

    def transform(self, document: Document) -> Document:
        try:
            # 尝试获取模型对应的编码
            encoding = tiktoken.encoding_for_model(self.config.openai_model)
        except KeyError:
            # 如果模型不被识别，使用默认编码（如 cl100k_base，这是 GPT-4 使用的编码）
            print(f"警告：无法为模型 {self.config.openai_model} 找到对应的分词器，使用默认分词器 cl100k_base")
            encoding = tiktoken.get_encoding("cl100k_base")
        
        content_token_size = len(encoding.encode(document.transformed_content))
        try:
            if content_token_size > self.config.openai_token_limit:
                raise TooManyTokensException(content_token_size, self.config.openai_token_limit)
        except Exception as e:
            print(e)
            return document
        return self.executeOpenAICall(document)
