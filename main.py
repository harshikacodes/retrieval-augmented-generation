from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = TextLoader("document loaders/notes.txt", encoding="utf-8")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [
        ("system", "yout are AI that summarizes the text"),
        ("human", "{data}")
    ]
)
model = ChatGroq(model="openai/gpt-oss-120b")

prompt = template.format_messages(data=docs[0].page_content)

result = model.invoke(prompt)

print(result.content)
