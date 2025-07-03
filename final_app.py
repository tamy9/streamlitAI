# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import chromadb
from transformers import pipeline
from pathlib import Path
import tempfile
from datetime import datetime

def add_custom_css():
    """Add custom styling to make the Queen app elegant and professional"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        color: #da9100;  /* gold-bronze Queen color */
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #1c1c1c, #000000);
        border-radius: 12px;
        font-family: 'Georgia', serif;
    }

    .stButton > button {
        color: #ffffff !important;           
        background-color: #a066d1 !important; 
        border: none;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        width: 100%;
    }

    .stTextInput>div>input {
        border-radius: 6px;
        padding: 0.75rem;
        border: 1px solid #6a0dad;
    }

    .stExpander {
        background-color: #f4ecf7;
        border-radius: 10px;
        font-family: 'Arial', sans-serif;
    }

    .metric-card {
        background: #fff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    body {
        background-image: url('https://favim.com/pd/p/orig/2018/12/10/queen-band-brian-may-john-deacon-Favim.com-6643310.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.85); /* light translucent overlay for visibility */
        padding: 2rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Document conversion imports (copy from your converter app)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice

# TODO: Copy your convert_to_markdown function here
def convert_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        pdf_opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_opts,
                    backend=DoclingParseV2DocumentBackend
                )
            }
        )
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext in [".doc", ".docx"]:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

    raise ValueError(f"Unsupported extension: {ext}")


# TODO: Copy your setup_documents function here  
def setup_documents():
    """
    This function creates our document database
    NOTE: This runs every time someone uses the app
    In a real app, you'd want to save this data permanently
    """
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="documents")
    except Exception:
        collection = client.create_collection(name="documents")
    
    return collection
    

# TODO: Copy your get_answer function here
def get_answer(collection, question):
    """
    This function retrieves the answer to a question from the document database
    """
    # Initialize the question-answering pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    
    # Retrieve relevant documents from the collection
    results = collection.query(query_texts=[question], n_results=5)
    
    # Check if we have any results
    if not results['documents'] or not results['documents'][0]:
        return "I don't have enough information to answer that question. Please upload some Queen-related documents first!"
    
    # Combine documents into a single context
    context = " ".join(results['documents'][0])
    
    # Check if context is too short
    if len(context.strip()) < 10:
        return "I don't have enough information to answer that question. Please upload some Queen-related documents first!"
    
    # Get the answer using the QA pipeline
    answer = qa_pipeline(question=question, context=context)
    
    return answer['answer']

# NEW: Function to handle uploaded files
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def reset_collection(client, collection_name: str):
    """Delete existing collection and create a new empty one"""
    
    try:
        # Delete existing collection
        client.delete_collection(name=collection_name)
        print(f"Deleted collection '{collection_name}'")
    except Exception as e:
        print(f"Collection '{collection_name}' doesn't exist or already deleted")
    
    # Create new empty collection
    new_collection = client.create_collection(name=collection_name)
    print(f"Created new empty collection '{collection_name}'")
    
    return new_collection




def add_text_to_chromadb(text: str, filename: str, collection_name: str = "documents"):
    """
    Add text to existing or new ChromaDB collection.
    Safe to call multiple times with same collection_name.
    """
    
    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,       
        chunk_overlap=100,     
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    # Initialize components (reuse if possible)
    if not hasattr(add_text_to_chromadb, 'client'):
        add_text_to_chromadb.client = chromadb.Client()
        add_text_to_chromadb.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        add_text_to_chromadb.collections = {}
    
    # Get or create collection
    if collection_name not in add_text_to_chromadb.collections:
        try:
            collection = add_text_to_chromadb.client.get_collection(name=collection_name)
        except:
            collection = add_text_to_chromadb.client.create_collection(name=collection_name)
        add_text_to_chromadb.collections[collection_name] = collection
    
    collection = add_text_to_chromadb.collections[collection_name]
    
    # Process chunks
    for i, chunk in enumerate(chunks):
        embedding = add_text_to_chromadb.embedding_model.encode(chunk).tolist()
        
        metadata = {
            "filename": filename,
            "chunk_index": i,
            "chunk_size": len(chunk)
        }
        
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[metadata],
            ids=[f"{filename}_chunk_{i}"]
        )
    
    print(f"Added {len(chunks)} chunks from {filename}")
    return collection


# Helper functions for the features
def convert_uploaded_files(uploaded_files):
    """Convert uploaded files to markdown"""
    converted_docs = []
    for file in uploaded_files:
        file_extension = Path(file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name
        
        text = convert_to_markdown(temp_file_path)
        converted_docs.append({
            'filename': file.name,
            'content': text
        })
    return converted_docs

def add_docs_to_database(collection, converted_docs):
    """Add documents to database"""
    count = 0
    for doc in converted_docs:
        add_text_to_chromadb(doc['content'], doc['filename'])
        count += 1
    return count

# Enhanced answer function with source tracking
def get_answer_with_source(collection, question):
    """Enhanced answer function that shows source document"""
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    
    docs = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]  # This tells us which document
    
    if not docs or min(distances) > 1.5:
        return "I don't have information about that topic.", "No source"
    
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    prompt = f"""Context information:
{context}

Question: {question}

Answer:"""
    
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    
    answer = response[0]['generated_text'].strip()
    
    # Extract source from best matching document
    best_source = ids[0].split('_doc_')[0]  # Get filename from ID
    
    return answer, best_source

# Document manager with delete option
def show_document_manager():
    """Display document manager interface"""
    st.subheader("🎶 Manage your Queen archive here")
    
    if not st.session_state.converted_docs:
        st.info("No documents uploaded yet.")
        return
    
    # Show each document with delete button
    for i, doc in enumerate(st.session_state.converted_docs):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"📄 {doc['filename']}")
            st.write(f"   Words: {len(doc['content'].split())}")
        
        with col2:
            # Preview button
            if st.button("Preview", key=f"preview_{i}"):
                st.session_state[f'show_preview_{i}'] = True
        
        with col3:
            # Delete button
            if st.button("Delete", key=f"delete_{i}"):
                # Remove from session state
                st.session_state.converted_docs.pop(i)
                # Rebuild database
                st.session_state.collection = setup_documents()
                add_docs_to_database(st.session_state.collection, st.session_state.converted_docs)
                st.rerun()
        
        # Show preview if requested
        if st.session_state.get(f'show_preview_{i}', False):
            with st.expander(f"Preview: {doc['filename']}", expanded=True):
                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                if st.button("Hide Preview", key=f"hide_{i}"):
                    st.session_state[f'show_preview_{i}'] = False
                    st.rerun()

# Search history
def add_to_search_history(question, answer, source):
    """Add search to history"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Add new search to beginning of list
    st.session_state.search_history.insert(0, {
        'question': question,
        'answer': answer,
        'source': source,
        'timestamp': str(datetime.now().strftime("%H:%M:%S"))
    })
    
    # Keep only last 10 searches
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[:10]

def show_search_history():
    """Display search history"""
    st.subheader("🕒 Recent searches")
    
    if 'search_history' not in st.session_state or not st.session_state.search_history:
        st.info("No searches yet.")
        return
    
    for i, search in enumerate(st.session_state.search_history):
        with st.expander(f"Q: {search['question'][:50]}... ({search['timestamp']})"):
            st.write("**Question:**", search['question'])
            st.write("**Answer:**", search['answer'])
            st.write("**Source:**", search['source'])

# Document statistics
def show_document_stats():
    """Show statistics about uploaded documents"""
    st.subheader("🎚️ Statistics about uploaded docs")
    
    if not st.session_state.converted_docs:
        st.info("No documents to analyze.")
        return
    
    # Calculate stats
    total_docs = len(st.session_state.converted_docs)
    total_words = sum(len(doc['content'].split()) for doc in st.session_state.converted_docs)
    avg_words = total_words // total_docs if total_docs > 0 else 0
    
    # Display in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", total_docs)
    
    with col2:
        st.metric("Total Words", f"{total_words:,}")
    
    with col3:
        st.metric("Average Words/Doc", f"{avg_words:,}")
    
    # Show breakdown by file type
    file_types = {}
    for doc in st.session_state.converted_docs:
        ext = Path(doc['filename']).suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    
    st.write("**File Types:**")
    for ext, count in file_types.items():
        st.write(f"• {ext}: {count} files")

# Enhanced UI with tabs
def create_tabbed_interface():
    """Create a tabbed interface for better organization"""
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎼 Upload", "🎙️ Ask questions", "🎶 Manage archive", "🎚️ Stats"])
    
    with tab1:
        st.header("🎼Upload & convert Queen documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if st.button("Chunk and store Queen docs🎸"):
            if uploaded_files:
                converted_docs = convert_uploaded_files(uploaded_files)
                if converted_docs:
                    num_added = add_docs_to_database(st.session_state.collection, converted_docs)
                    st.session_state.converted_docs.extend(converted_docs)
                    st.success(f"Added {num_added} documents!")
    
    with tab2:
        st.header("Curious about Queen? Type your question here!🎧")
        
        if st.session_state.converted_docs:
            question = st.text_input("Your question:")
            
            if st.button("Rock me the answer!🎸"):
                if question:
                    with st.spinner("🎤 Warming up Freddie’s mic..."):
                
                        answer, source = get_answer_with_source(st.session_state.collection, question)
                    
                    st.write("**Answer:**")
                    st.write(answer)
                    st.write(f"**Source:** {source}")
                    st.success("👑 The Queen archives have spoken – enjoy your insight!")
                    
                    # Add to history
                    add_to_search_history(question, answer, source)
        else:
            st.info("Upload documents first!")
        
        # Show recent searches
        show_search_history()
    
    with tab3:
        show_document_manager()
    
    with tab4:
        show_document_stats()

# MAIN APP
def main():
    # Initialize session state
    if 'converted_docs' not in st.session_state:
        st.session_state.converted_docs = []
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'collection' not in st.session_state:
        st.session_state.collection = setup_documents()
    
    add_custom_css()
    st.markdown('<h1 class="main-header" style="color:white;">👑Queen: The Rock Royalty🎸</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
    <h3 style='color:#6a0dad;'>🎶 Welcome to the Queen Knowledge Hub! 🎶</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center;'>
    <span style="color:#cd7f32;"><em>Explore the remarkable journey, groundbreaking music, and enduring legacy of one of the greatest bands in rock history.</em></span>
    </div>
    """, unsafe_allow_html=True)

    # Add custom button styling
    st.markdown("""
        <style>
            .stButton>button {
                color: black !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Show about section
    with st.expander("About this Queen Q&A System"):
        st.markdown("""
            <div style='color:purple; font-size:16px;'>
                <p>This system is built to answer your questions about the legendary rock band Queen, including:</p>
                <ul>
                    <li>The history and rise of Queen</li>
                    <li>Detailed profiles of each band member</li>
                    <li>Albums, songs, and major discography highlights</li>
                    <li>Awards, nominations, and career milestones</li>
                    <li>Musical style, influences, and genre-blending innovations</li>
                </ul>
                <p>Dive in and discover the stories, sounds, and legacy behind one of the greatest bands in rock history.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Create the tabbed interface with all features
    create_tabbed_interface()

if __name__ == "__main__":
    main()
