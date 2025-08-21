from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr

# Load and process PDF
def load_pdf(pdf_file):
    raw_text = ""
    with pdfplumber.open(pdf_file.name) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
    return raw_text

# Build retriever
def build_retriever_from_text(raw_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Load LLM
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    return HuggingFacePipeline(pipeline=pipe)

# Initialize global objects
retriever = None
qa_chain = None
llm = load_llm()

# Gradio interface logic
def setup_qa(pdf_file):
    global retriever, qa_chain
    text = load_pdf(pdf_file)
    retriever = build_retriever_from_text(text)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return "✅ PDF uploaded and processed. You can now ask questions."

def ask_question(query):
    if not qa_chain:
        return "❗ Please upload a PDF first."
    result = qa_chain.invoke({"query": query})
    return result["result"]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 PDF ChatBot with LangChain + Chroma")
    with gr.Row():
        pdf_file = gr.File(label="Upload PDF")
        upload_button = gr.Button("Process PDF")
    status = gr.Textbox(label="Status")

    with gr.Row():
        question = gr.Textbox(label="Ask a Question")
        answer = gr.Textbox(label="Answer")
        ask_button = gr.Button("Ask")

    upload_button.click(fn=setup_qa, inputs=pdf_file, outputs=status)
    ask_button.click(fn=ask_question, inputs=question, outputs=answer)

demo.launch()
