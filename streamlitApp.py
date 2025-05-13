import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re

# Set page configuration
st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="Icon.jpg",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main {
        background-color: #000000;
        padding: 2rem;
        border-radius: 15px;
    }
    .stTextInput > div > div > input {
        background-color: #E8C999 !important;
        color: black !important;
        border-radius: 10px;
        padding: 12px;
        border: 2px solid #4CAF50;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #F8EEDF;
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.3);
    }
    .stButton > button {
        background-color: #8E1616;  /* Deep purple */
        color: #ffffff;
        font-weight: bold;
        padding: 10px 24px;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background-color: #8A2D3B;  
        transform: scale(1.05);
        cursor: pointer;
        color: #ffffff;
    }
    .stSpinner {
        color: #4CAF50;
        font-style: italic;
    }
    h1, h2, h3 {
        color: #2E7D32;
    }
    .answer-box {
        background-color: #F8EEDF;       
        color: #0d1b2a;                  
        border-left: 4px solid #F8EEDF;  
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .custom-title {
        color: #F8EEDF;
        font-size: 50px;
        font-weight: bold;
        margin-bottom: 0;
    }
    .custom-subtitle {
        color: #F8EEDF;
        font-size: 30px;
        margin-top: 10px;
        font-weight: bold;
    }
    .custom-subtitle2 {
        color: #F8EEDF;
        font-size: 20px;
        margin-top: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Utility function to validate YouTube video ID
def validate_youtube_id(video_id):
    """Validate YouTube video ID format"""
    pattern = r'^[a-zA-Z0-9_-]{11}$'
    return re.match(pattern, video_id) is not None

# Main Streamlit app
def main():
    # Initialize session state variables if not exists
    if 'transcript_loaded' not in st.session_state:
        st.session_state['transcript_loaded'] = False
    if 'last_video_id' not in st.session_state:
        st.session_state['last_video_id'] = ''
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []

    # Display Title and Subtitle
    st.markdown('<div class="custom-title">🎥 YouTube Video Insight Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-subtitle">Get insights from YouTube video transcripts using AI!</div>', unsafe_allow_html=True)

    # Video ID input with validation
    st.markdown('<div class="custom-subtitle2">🎥 Enter the YouTube Video Id</div>', unsafe_allow_html=True)
    video_id = st.text_input(
        "", 
        value=st.session_state.get('last_video_id', ''),
        placeholder="e.g. Gfr50f6ZBvo",
        help="Find the video ID in the YouTube URL after 'v='"
    )

    # Fetch and process transcript
    col1, col2 = st.columns([3, 1])
    with col1:
        load_transcript = st.button("Load Transcript", use_container_width=True)
    
    if load_transcript or (video_id and video_id != st.session_state['last_video_id']):
        # Validate video ID
        if not video_id:
            st.error("❌ Please enter a valid YouTube Video ID")
            return
        
        if not validate_youtube_id(video_id):
            st.error("❌ Invalid YouTube Video ID format")
            return

        # Attempt to fetch transcript
        with st.spinner("🔍 Fetching transcript..."):
            try:
                # Fetch transcript
                transcriptList = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                transcript = " ".join(chunk["text"] for chunk in transcriptList)

                # Split transcript into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])

                # Generate embeddings
                with st.spinner("🔎 Building knowledge base..."):
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

                    # Configure prompt and LLM
                    prompt = PromptTemplate(
                        template="""You are a helpful AI assistant analyzing a YouTube video transcript.
                        Use ONLY the provided context to answer the question.
                        If the context doesn't contain enough information, respond with "I don't have enough information from the transcript to answer this."
                        
                        Context:
                        {context}
                        
                        Question: {question}
                        """,
                        input_variables=["context", "question"]
                    )

                    # Use Google's Gemini model
                    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

                    # Setup chain
                    def format_docs(retrieveDocs):
                        return "\n\n".join(doc.page_content for doc in retrieveDocs)

                    parallelChain = RunnableParallel({
                        'context': retriever | RunnableLambda(format_docs),
                        'question': RunnablePassthrough()
                    })

                    parser = StrOutputParser()
                    mainChain = parallelChain | prompt | llm | parser

                    # Store in session state for further use
                    st.session_state['transcript_loaded'] = True
                    st.session_state['retriever'] = retriever
                    st.session_state['mainChain'] = mainChain
                    st.session_state['last_video_id'] = video_id
                    
                    st.success("✅ Transcript loaded successfully!")

            except TranscriptsDisabled:
                st.error("❌ This video has no captions available.")
            except Exception as e:
                st.error(f"⚠️ Error fetching transcript: {e}")

    # Question asking section
    if st.session_state.get('transcript_loaded', False):
        st.markdown("### 💬 **Ask a Question about the Video**")

        # Custom function to handle question submission
        def submit_question():
            # Get the current query from session state
            user_query = st.session_state.get('query_input', '').strip()
            
            # Clear the input after submission
            st.session_state['query_input'] = ''
            
            # Validate query
            if not user_query:
                st.warning("Please enter a question!")
                return

            with st.spinner("🤖 Generating response..."):
                try:
                    # Retrieve and generate answer
                    answer = st.session_state['mainChain'].invoke(user_query)
                    
                    # Add to conversation history
                    st.session_state['conversation_history'].append({
                        'question': user_query,
                        'answer': answer
                    })
                    
                    # Display response with styling
                    st.markdown("### **🧠 AI Analysis**")
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Failed to generate answer: {e}")

        # Question input with key to enable return key submission
        user_query = st.text_input(
            "Your question", 
            placeholder="e.g. What is the main topic?", 
            key="query_input",
            on_change=submit_question  # This allows submission by pressing Enter
        )

        # Optional explicit submit button
        st.button("Get Answer", on_click=submit_question, use_container_width=True)

        # Show conversation history
        if st.session_state['conversation_history']:
            st.markdown("### **📜 Conversation History**")
            for idx, entry in enumerate(reversed(st.session_state['conversation_history']), 1):
                st.markdown(f"#### Question {len(st.session_state['conversation_history']) - idx + 1}")
                st.markdown(f"**Q:** {entry['question']}")
                st.markdown(f'<div class="answer-box">{entry["answer"]}</div>', unsafe_allow_html=True)




if __name__ == "__main__":

    # Note: Replace with your actual Google API key securely
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBToCHAjcOL6ce2qdYfBmzXH6nJI7TOJnI"
    main()
    
    st.markdown("""
<div style="text-align:center;position: relative;
        bottom: 0;
        width: 100%;;padding: 10px;margin-top:160px;font-size:16px;">
    <hr>
    🎬 Developed by Soham Kale | Powered by LangChain & Gemini 🧠
</div>
""", unsafe_allow_html=True)
