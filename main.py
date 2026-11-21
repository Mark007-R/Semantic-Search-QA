import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Groq model used for answers. Groq retires model IDs over time; current
# ones are listed at https://console.groq.com/docs/models
GROQ_MODEL = "openai/gpt-oss-120b"

def initialize_system():
    """
    Initialize the RAG system by loading text, creating embeddings,
    and setting up the vector store.
    """
    print("=" * 60)
    print("Q&A System - Initializing...")
    print("=" * 60)
    
    # Step 1: Load the speech text
    print("\n[1/6] Loading speech text...")
    try:
        loader = TextLoader('speech.txt', encoding='utf-8')
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} document(s)")
    except FileNotFoundError:
        print("✗ Error: speech.txt not found!")
        print("Please ensure speech.txt is in the same directory as main.py")
        return None, None, None
    
    # Step 2: Split text into chunks
    print("\n[2/6] Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(
        separator=". ",
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} text chunks")
    
    # Step 3: Create embeddings
    print("\n[3/6] Loading embedding model...")
    print("(First run may take time to download ~90MB model)")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✓ Embedding model loaded")
    
    # Step 4: Create vector store
    print("\n[4/6] Creating vector store with ChromaDB...")
    vectorstore = Chroma(
        collection_name="ambedkar_speech",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    # chroma_db is kept on disk between runs. Start from an empty collection
    # each time; otherwise every run adds the same chunks again and the
    # retriever returns duplicate copies of one chunk.
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)
    print(f"✓ Vector store created with {len(chunks)} embeddings")
    
    # Step 5: Initialize LLM
    print("\n[5/6] Initializing Groq LLM...")
    groq_api_key = os.getenv("GROQ_API_KEY")
    # print(groq_api_key)
    
    if not groq_api_key:
        print("✗ Error: GROQ_API_KEY not found!")
        print("Please set your Groq API key in .env file")
        print("Format: GROQ_API_KEY=your_api_key_here")
        return None, None, None
    
    try:
        # This model reasons before it answers, and the reasoning counts
        # against max_tokens, so the cap leaves room for both.
        llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.3,
            api_key=groq_api_key,
            max_tokens=1500
        )
        print(f"✓ Groq LLM initialized ({GROQ_MODEL})")
    except Exception as e:
        print(f"✗ Error initializing Groq: {str(e)}")
        print("Please check your API key and internet connection")
        return None, None, None
    
    # Step 6: Create RetrievalQA chain
    print("\n[6/6] Setting up RAG pipeline...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        verbose=False
    )
    print("✓ RAG pipeline ready!")
    
    print("\n" + "=" * 60)
    print("System initialized successfully!")
    print("=" * 60)
    
    return qa_chain, vectorstore, embeddings


def display_retrieved_context(source_docs):
    """
    Display the retrieved context chunks with their content.
    """
    print("\n📚 Retrieved Context:")
    print("-" * 60)
    for i, doc in enumerate(source_docs, 1):
        print(f"\n[Chunk {i}]")
        print(f"{doc.page_content}")
    print("-" * 60)


def ask_question(qa_chain, question):
    """
    Ask a question and get an answer with retrieved context.
    """
    print(f"\n❓ Question: {question}")
    
    try:
        # Get response from RAG chain
        result = qa_chain.invoke({"query": question})
        
        # Display retrieved context
        if result.get("source_documents"):
            display_retrieved_context(result["source_documents"])
        
        # Display answer
        print("\n💡 Answer:")
        print("-" * 60)
        print(result["result"])
        print("-" * 60)
        
    except Exception as e:
        print(f"\n✗ Error processing question: {str(e)}")


def interactive_mode(qa_chain):
    """
    Run the system in interactive mode for continuous Q&A.
    """
    print("\n" + "=" * 60)
    print("Interactive Q&A Mode")
    print("=" * 60)
    print("Ask questions about Dr. B.R. Ambedkar's speech.")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("\nExample questions:")
    print("  - What is the real remedy according to the speech?")
    print("  - What does the text say about social reform?")
    print("  - What is the real enemy mentioned in the speech?")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n🔍 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', '']:
                print("\nThank you for using AmbedkarGPT! Goodbye! 👋")
                break
            
            ask_question(qa_chain, question)
            
        except (KeyboardInterrupt, EOFError):
            # EOFError: input closed (Ctrl+Z / Ctrl+D, or piped input ran out).
            # Without this the loop below would print the error forever.
            print("\n\nSession interrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


def main():
    """
    Main function to run the Q&A System.
    """
    # Initialize the system
    qa_chain, vectorstore, embeddings = initialize_system()
    
    if qa_chain is None:
        print("\n✗ System initialization failed. Please fix the errors and try again.")
        return
    
    # Run interactive Q&A mode
    interactive_mode(qa_chain)


if __name__ == "__main__":
    main()