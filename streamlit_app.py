
from pinecone import Pinecone, ServerlessSpec  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA 
#from langchain.vectorstores import Pinecone  # ✅ Correct Import


groq_api_key = "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"


  

 
PINECONE_API_KEY = "pcsk_2vt4pm_N1aMKkEqkKnVdvqA8V4yp8PLpAQnV45Febmc5ThrApyMp8WUsGtHtev4NJj7uJn"  # ✅ Correct
print(PINECONE_API_KEY)
#PINECONE_API_KEY = os.environ('PINECONE_API_KEY')  # Set this in your environment
PINECONE_ENV = "us-east1-aws"  # Change based on your Pinecone setup
INDEX_NAME = "pdf-knowledgebase"  # Set your desired index name



pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# ✅ Set Up Vectorstore
#vectorstore = Pinecone(index, embeddings.embed_query)
#vectorstore = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
from langchain_pinecone import PineconeVectorStore
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

from langchain_groq import ChatGroq
##llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
##llm = ChatGroq(groq_api_key=groq_api_key, model_name="Deepseek-R1-Distill-Llama-70b")
import streamlit as st
# --- 🎨 Streamlit UI ---
st.markdown("<h1 class='title'>AI Chatbot</h1>", unsafe_allow_html=True)

# ✅ Dropdown to select LLM Model
model_options = {
    "Mixtral (8x7B)": "mixtral-8x7b-32768",
    "DeepSeek (Llama 70B)": "Deepseek-R1-Distill-Llama-70b",
    "Gemma (9B IT)": "Gemma2-9b-it"
}
selected_model_name = st.selectbox("Select an AI Model:", list(model_options.keys()))

# Store selected model in session_state
if "selected_model" not in st.session_state or st.session_state.selected_model != model_options[selected_model_name]:
    st.session_state.selected_model = model_options[selected_model_name]
    st.session_state.chat_history = []  # Reset chat history if model changes

# Initialize LLM with selected model
llm = ChatGroq(groq_api_key=groq_api_key, model_name=st.session_state.selected_model)

#llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-it")


from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)



st.markdown("""
<style>
.title {
    color: #FF69B4; /* Pink color */
    font-size: 30px;
    font-weight: bold;
}
.chat-message {
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    max-width: 70%;
}
.chat-message.user {
    background-color: #f5bad7;
    margin-left: 30%;
}
.chat-message.bot {
    background-color: #FF69B4;
    margin-right: 30%;
}
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt_input = st.text_input("Input your prompt here")
if prompt_input:
    st.session_state.chat_history.append({"role": "user", "content": prompt_input})
    response = retrieval_chain.invoke({"input": prompt_input})
    print(response)
    st.session_state.chat_history.append({"role": "bot", "content": response['answer']})

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot">{message["content"]}</div>', unsafe_allow_html=True)
