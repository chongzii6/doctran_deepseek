from doctran import Doctran

# 使用 DeepSeek 模型 (仅支持 V2.5)
doctran = Doctran(
    openai_api_key="sk-fe19314cfe7d424d9c8ec7c52f3dcd88",
    openai_model="qwen-turbo",
    openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # DeepSeek API 端点
)


mess_txt = '''
<doc type="Confidential Document - For Internal Use Only">
<metadata>
<date> &#x004A; &#x0075; &#x006C; &#x0079; &#x0020; &#x0031; , &#x0032; &#x0030; &#x0032; &#x0033; </date>
<subject> Updates and Discussions on Various Topics; </subject>
</metadata>
<body>
<p>Dear Team,</p>
<p>I hope this email finds you well. In this document, I would like to provide you with some important updates and discuss various topics that require our attention. Please treat the information contained herein as highly confidential.</p>
<section>
<header>Security and Privacy Measures</header>
<p>As part of our ongoing commitment to ensure the security and privacy of our customers' data, we have implemented robust measures across all our systems. We would like to commend John Doe (email: john.doe&#64;example.com) from the IT department for his diligent work in enhancing our network security. Moving forward, we kindly remind everyone to strictly adhere to our data protection policies and guidelines. Additionally, if you come across any potential security risks or incidents, please report them immediately to our dedicated team at security&#64;example.com.</p>
</section>
<section>
<header>HR Updates and Employee Benefits</header>
<p>Recently, we welcomed several new team members who have made significant contributions to their respective departments. I would like to recognize Jane Smith (SSN: &#x0030; &#x0034; &#x0039; - &#x0034; &#x0035; - &#x0035; &#x0039; &#x0032; &#x0038;) for her outstanding performance in customer service. Jane has consistently received positive feedback from our clients. Furthermore, please remember that the open enrollment period for our employee benefits program is fast approaching. Should you have any questions or require assistance, please contact our HR representative, Michael Johnson (phone: &#x0034; &#x0031; 
...
'''
# 解析文档
document = doctran.parse(content=mess_txt)

import json

# transformed_document = document.summarize(token_limit=100).execute()
# print(transformed_document.transformed_content)


# transformed_document = document.refine(topics=['marketing', 'company events']).execute()
# print(transformed_document.transformed_content)

# transformed_document = document.interrogate().execute()
# print(json.dumps(transformed_document.extracted_properties, indent=2))

from doctran import ExtractProperty

properties = [
        # ExtractProperty(
            # name="millenial_or_boomer", 
            # description="A prediction of whether this document was written by a millenial or boomer",
            # type="string",
            # enum=["millenial", "boomer"],
            # required=True
        # ),
        # ExtractProperty(
            # name="as_gen_z", 
            # description="The document summarized and rewritten as if it were authored by a Gen Z person",
            # type="string",
            # required=True
        # ),
        ExtractProperty(
            name="contact_info", 
            description="A list of each person mentioned and their contact information",
            type="array",
            items={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the person"
                    },
                    "contact_info": {
                        "type": "object",
                        "properties": {
                            "phone": {
                                "type": "string",
                                "description": "The phone number of the person"
                            },
                            "email": {
                                "type": "string",
                                "description": "The email address of the person"
                            }
                        }
                    }
                }
            },
            required=True
        )
]

# properties = [ExtractProperty(
    # name="millenial_or_boomer", 
    # description="A prediction of whether this document was written by a millenial or boomer",
    # type="string",
    # enum=["millenial", "boomer"],
    # required=True
# )]

transformed_document = document.extract(properties=properties).execute()
print(json.dumps(transformed_document.extracted_properties, indent=2))

