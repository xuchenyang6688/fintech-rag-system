import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


class FinancialRAGSystem:
    def __init__(self):
        api_key = os.getenv("ZHIPUAI_API_KEY")
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.llm = ChatZhipuAI(api_key=api_key, temperature=0.1, model="glm-4")
        self.vectorstore = None
        self.retriever = None

    def process_document(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", ";", ",", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            chunks, self.embeddings, persist_directory="./chroma_db"
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        print(f"Processed {len(chunks)} chunks from PDF")
        return self.retriever

    def create_chain(self):
        """Create and return the RAG chain"""
        if not self.retriever:
            raise ValueError("Please process documents first using process_document()")

        template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        
        Answer: """

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain


if __name__ == "__main__":
    rag_system = FinancialRAGSystem()
    chain = rag_system.process_document(
        "../../../docs/caching-at-scale-with-redis-updated-2021-12-04.pdf"
    )
    chain = rag_system.create_chain()
    result = chain.invoke(
        "List the different kinds of the cache patterns,like cache-aside. For each cache pattern, output its name and an explanation of this pattern within 100 words."
    )

    print("Answer:", result)
