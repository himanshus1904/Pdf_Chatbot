# PDF Chatbot
This project aims to create an interactive chatbot capable of answering questions related to uploaded PDF documents 
leveraging Langchain, Google Gemini Pro and Streamlit for a simple UI
# Features
* PDF Upload: Users can upload multiple PDF documents(can be unrelated).
* Chatbot Interface: The chatbot provides responses to any question/ask based on the content of the uploaded PDFs.
* Streamlit UI: The project is built using Streamlit, providing a user-friendly interface for interacting with the chatbot.
# Process
* Gets the required Pdfs from the user - Can be any number of PDF files(each <250 mb)
* Creates a raw text combining all the pages of all the files
* Splits the text into chunks of 10000 with an overlap of 1000 words so that context is not lost in boundaries
* Stores the chunks into a vector store and is now Ready for user questions
* Forms the prompt using the user input and the context
* Searches the vector store using similarity search and gets the best possible reply
# Packages Used
* **streamlit** : Used for creating interactive web applications with Python. It allows you to easily build and share
* data-driven applications.
* **PyPDF2** : PyPDF2 is a Python library for working with PDF files. It allows you to extract text, merge, split, crop 
and manipulate PDF documents programmatically.
* **langchain.text_splitter.RecursiveCharacterTextSplitter**: Used for text processing tasks. It helps in 
splitting text into smaller chunks using recursive character-based splitting.
* **langchain_google_genai.GoogleGenerativeAIEmbeddings**: This allows you to generate embeddings for text data 
using pre-trained models from Google's Generative AI.
* **google.generativeai**: This package provides access to Google's Generative AI models.
* **langchain_community.vectorstores**: It is used for storing and querying vector embeddings.
* **langchain_google_genai.ChatGoogleGenerativeAI**: This allows you to generate conversational responses using 
pre-trained chatbot models from Google.
* **langchain.chains.question_answering.load_qa_chain**: This helps to build systems that can answer questions 
based on given input.
*  **langchain.prompts.PromptTemplate**: This provides templates for generating prompts tailored to specific tasks
or scenarios.
# Setup Instructions
1. Clone the Repository: git clone <repo-url>
2. Create a virtual env with python>=3.9
3. Install the requruirements.txt
```python
pip install requirements.txt
```
4. Run the streamlit app
```python
streamlit run app.py
```
5. Upload multiple/single PDF(s) and submit
6. All set to ask questions

