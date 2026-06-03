from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
import streamlit as st
from time import sleep

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=api_key)
api_key=st.secrets["GOOGLE_API_KEY"]

if "vector_db" not in st.session_state:
    st.session_state.vector_db=None


if "messages" not in st.session_state:
    st.session_state.messages=[]


def document_processs(path):

## document loading
    loader = PyPDFLoader(path)
    docs = loader.load()
# print(len(docs))



    ## splitting
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs=text_splitter.split_documents(docs)
    print(len(docs))





    ## embedding and vector store
    embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview",google_api_key=api_key)                                       
    vector_db=InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings)
    st.session_state.vector_db=vector_db
    st.session_state.document_uploaded=True










# print(answer.content)
st.subheader("📄 Document Q&A Chatbot - Ask Anything")
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded=False
#### document upload
if not st.session_state.document_uploaded:
    file=st.file_uploader(label="Select your pdf file", type="pdf")
    if file:
        with open("uploaded_document.pdf","wb") as f:
            f.write(file.getvalue())
        
        with st.spinner("Processing document..."):
            document_processs("./uploaded_document.pdf")

        st.markdown("Document uploaded successfully! Please ask your question in the chat below.")
        sleep(2)
        st.rerun()

### chat ui
if  st.session_state.document_uploaded and st.session_state.vector_db :

    for oneMessage in st.session_state.messages:
        role=oneMessage["role"]
        content=oneMessage["content"]
        st.chat_message(role).markdown(content)

    query=st.chat_input("Ask a question about the document")
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        documents=st.session_state.vector_db.similarity_search(query, k=2)
        context=""
        for doc in documents:
            context=context+doc.page_content +"\n\n\n"
        prompt=f"""
You are a helpful assistant for answering questions related to the document. Use the below context to answer the question. If you don't know the answer, say you don't know. Always use all the context available to you. Context: {context} Question: {query}

"""
        result=llm.invoke(prompt)
        st.session_state.messages.append({"role":"assistant","content":result.content})
        st.chat_message("assistant").markdown(result.content)
    




