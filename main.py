from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = PyPDFLoader("document loaders/GRU.pdf")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [
        ("system", "yout are AI that summarizes the text"),
        ("human", "{data}")
    ]
)
model = ChatGroq(model="openai/gpt-oss-120b")

prompt = template.format_messages(data=docs)

result = model.invoke(prompt)

print(result.content)
