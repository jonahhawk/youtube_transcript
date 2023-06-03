from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import re
import time
from colorama import Fore, Style

start_time = time.time()
# get the data with a Loader
url = "https://www.youtube.com/watch?v=2xNzB7xq8nk&ab_channel=DavidShapiro~AI"
loader = YoutubeLoader.from_youtube_url(url)
transcript = loader.load()



# split the transcript into chunks with a Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(transcript)

# create a vectorstore database
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

# create a query to search for similar chunks
human_query = "What does the speaker say about Pinecone?"
similar_chunks = db.similarity_search(human_query, k=4)
db_creation_time = time.time()

# print the content of the similar chunks
for chunk in similar_chunks:
    print(Fore.GREEN + f"Similar chunk: {chunk.page_content}" + Style.RESET_ALL)
print("Sending query to OpenAI API...")
# get the content of the similar chunks and join them into a single string
chunks_page_content = " ".join([d.page_content for d in similar_chunks])

# create an instance of the chat model
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

# create a template into which the similar_chunks will be inserted
prompt_template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {content}
        Only use the factual information from the transcript to answer the question.
        If you don't have enough information to answer the question, say "I don't know".
        Your answers should be verbose and detailed.
        """
# create the system prompt and human prompt
system_message_prompt = SystemMessagePromptTemplate.from_template(prompt_template)

human_template = "Answer this question: {question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# create an object for the chat prompt
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# create a chain with the chat model and the chat prompt
chain = LLMChain(llm=chat, prompt=chat_prompt)
pre_response = time.time()

# run the chain to generate a response from the OpenAI API
response = chain.run(content=chunks_page_content, question=human_query)

post_response = time.time()
# remove newlines from the response
response = re.sub(r"\n", " ", response)

# print the query, response and report some timing information
print(Fore.RED + human_query + Style.RESET_ALL)
print(Fore.MAGENTA + response + Style.RESET_ALL)
print(Fore.CYAN + f"Embedding, db creation time: {format((db_creation_time - start_time), '.2f')}" + Style.RESET_ALL)
print(Fore.CYAN + f"Semantic search time: {format((pre_response - db_creation_time), '.8f')}" + Style.RESET_ALL)
print(Fore.CYAN + f"Response time to/from OpenAI API:{format((post_response - pre_response), '.2f')}")
print(f"Total time: {format((post_response - start_time), '.2f')}" + Style.RESET_ALL)
