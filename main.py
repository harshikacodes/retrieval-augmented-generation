from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# retrievers 

template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are an AI that summarizes the text"),
        ("human", "{data}")
    ]
)
model = ChatGroq(model="openai/gpt-oss-120b")

