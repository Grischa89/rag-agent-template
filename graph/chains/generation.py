from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0, model=os.environ["model"])
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()