from langchain_community.retrievers import ArxivRetriever

# create retriever
retriever = ArxivRetriever(
    load_max_docs=2, # num of papers to retrieve
    load_all_available_meta=True
)

# query arxiv
docs = retriever.invoke("large language models")

# print result
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title: ", doc.metadata.get("Title"))
    print("Author: ", doc.metadata.get("Author"))
    print("Summary: ", doc.page_content[:500])