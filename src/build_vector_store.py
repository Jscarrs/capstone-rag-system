from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


def main():
    print("Loading book text...")
    loader = TextLoader("data/art_of_war.txt", encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} total chunks.")

    # 🔹 LIMIT chunks for prototype (performance)
    chunks = chunks[:300]
    print(f"Using {len(chunks)} chunks for vector store prototype.")

    print("Creating embeddings using local Ollama (this may take several minutes)...")
    embeddings = OllamaEmbeddings(model="llama3")

    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("Saving vector store to disk...")
    vectorstore.save_local("vector_store/art_of_war")

    print("Vector store created and saved successfully.")


if __name__ == "__main__":
    main()
