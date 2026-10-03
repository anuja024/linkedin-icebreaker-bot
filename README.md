# LinkedIn Icebreaker Bot 🤖

An AI-powered RAG application that analyzes LinkedIn profile data, generates interesting profile-based facts, and lets users ask questions about the person through a conversational interface.

The project uses **LlamaIndex** for document processing and semantic retrieval, **Hugging Face embeddings** for local vector representations, **Groq's GPT-OSS-20B** for response generation, and **Gradio** for the web interface.

---

## ✨ Features

- Extract LinkedIn profile data using ProxyCurl API or built-in mock data
- Split profile information into semantic chunks
- Generate local embeddings using Hugging Face
- Store and retrieve profile information using LlamaIndex
- Generate 3 interesting facts about a profile
- Ask questions about the profile using RAG
- Maintain a profile-specific conversational session
- Interactive Gradio web interface
- Supports local development without requiring IBM Cloud or watsonx.ai

---

## 🧠 How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

```text
LinkedIn Profile
       ↓
Profile Data Extraction
       ↓
Data Chunking
       ↓
Hugging Face Embeddings
       ↓
LlamaIndex Vector Index
       ↓
Semantic Retrieval
       ↓
Relevant Profile Context
       ↓
Groq GPT-OSS-20B
       ↓
Profile-Based Response