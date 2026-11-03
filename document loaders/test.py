from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 100,
    chunk_overlap=10
)

data = TextLoader("document loaders/notes.txt", encoding="utf-8")
docs = data.load()

chunks = splitter.split_documents(docs)

print(len(chunks));

for i in chunks:
    print(i.page_content)
    print()