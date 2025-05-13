
# 🎥 YouTube Video Insight Chatbot
### APP -[https://youtube-chatbot-sk28.streamlit.app]
Unlock meaningful insights from YouTube videos using the power of AI!  
This chatbot fetches video transcripts and allows users to ask natural language questions about the video content, giving precise and context-aware responses using advanced language models.

---

## 💡 Overview

The **YouTube Video Insight Chatbot** is an interactive web application built with **Streamlit** that lets you:

- 🧠 Ask questions about a YouTube video by just providing its **Video ID**
- 📄 Automatically fetch and analyze the **transcript**
- 🔍 Use semantic search (via vector embeddings) to understand video content deeply
- 🤖 Get smart and accurate answers from **Google Gemini (via LangChain)**

Whether you're summarizing lectures, analyzing interviews, or extracting key points from podcasts — this tool turns any transcript into an interactive knowledge base.

---

## 🚀 Features

- 🎬 **Video Transcript Retrieval**: Supports English captions using YouTubeTranscriptAPI
- 🧩 **Chunking & Embeddings**: Splits large transcripts and creates vector embeddings using `HuggingFace`
- 🧠 **Contextual Q&A**: Uses `LangChain` to query the relevant transcript context
- ✨ **Gemini Integration**: Leverages `Gemini 2.0 Flash` for fast, accurate LLM responses
- 💬 **Conversation History**: Keeps track of your queries for easy reference

---

## 🖼️ Demo

| Get Answers | Ask Questions | See History | 
|-----------------|----------------|--------------|
| ![Image1](https://drive.google.com/uc?export=view&id=1WvNFptVwAsBlsOUDf0EtLLJPKPvzYuLP) | ![Image2](https://drive.google.com/uc?export=view&id=1-0zp8PtTNmjun6Sqjweg5VCYkKGE6-Hd) | ![Image3](https://drive.google.com/uc?export=view&id=1yTc7qislYrd1mn2grIjSzOJIUyEbKhpG) 

---

## 🤖 Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **Google Generative AI (Gemini)**
- **HuggingFace Embeddings**
- **FAISS Vector Store**
- **YouTubeTranscriptAPI**

---

## 📌 Notes

- Only works with YouTube videos that have **English transcripts enabled**
- The video ID is the string after `v=` in a YouTube URL (e.g., `Gfr50f6ZBvo`)

---

## 📬 Credits

Developed by **Soham Kale**  
Powered by **LangChain** & **Google Gemini**

