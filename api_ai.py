from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from pinecone import ServerlessSpec
from pathlib import Path

import requests

import logging
import datetime
from supabase import create_client, Client
from pinecone import Pinecone
import time
import openai
import pandas as pd 
import uuid
import os

# defining pinecone 
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone( api_key=PINECONE_API_KEY)

url_storage_part = os.getenv('url_storage_part')

# defining supabase

url: str = os.getenv('SUPA_URL')
key: str = os.getenv('SUPA_KEY')
supabase: Client = create_client(url, key)

### Defining OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Functions to keep track of RAG updates 

def create_rag_supabase(index_name):
    supabase.table("rag_update").insert({"created_at": datetime.datetime.now().isoformat(), 
             "updated_at": datetime.datetime.now().isoformat(),
             'type':'channel',
             "index_id": index_name, 
            }).execute()
    print(f"RAG {index_name} created in supabase")
    
def upsdate_rag_supabase(index_name):
    supabase.table("rag_update").update({"updated_at": datetime.datetime.now().isoformat()}).eq("index_id", index_name).execute()

def delete_rag_supabase(index_name):
    supabase.table("rag_update").delete().eq("index_id", index_name).execute()

def get_rag_supabase(index_name):
    resp = None
    rags_info =pd.DataFrame( supabase.table("rag_update").select("*").execute().data)
    if len(rags_info)==0:
        return None
    rags_info['created_at'] = pd.to_datetime(rags_info['created_at'])
    rags_info['updated_at'] = pd.to_datetime(rags_info['updated_at'])
    rag_records = rags_info[rags_info['index_id'] == index_name]
    if not rag_records.empty:
        resp = rag_records.to_dict('records')[0]
    return resp

### ETL functions 

def download_file(url, download_dir="downloads"):
    os.makedirs(download_dir, exist_ok=True)
    filename = os.path.join(download_dir, url.split("/")[-1])
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        return filename
    else:
        raise Exception(f"Failed to download file: {url}")
    

### downloading the files 
def folder_files(list_urls, folder_name):
    if len(list_urls)>0:
        for el in list_urls:
            download_file(el, folder_name)
    else:
        print('no files')
    


def load_pdf_to_pinecone(index_name, type):
    etl_history = get_rag_supabase(index_name)
    if etl_history is not None:
        start_date = etl_history['updated_at']
    else:
        start_date = None

    if type == 'channel':
        files_info =pd.DataFrame( supabase.table("files").select("*").execute().data)
        filtered_df = files_info[files_info["storage_path"].str.startswith(f"channel/{index_name}/")]
        filtered_df = filtered_df[filtered_df["storage_path"].str.endswith(".pdf", ".docx")]
        filtered_df['created_at'] = filtered_df['created_at'].apply(lambda x : pd.to_datetime(x).tz_localize(None))
        if start_date is not None:
            filtered_df = filtered_df[filtered_df['created_at'] > start_date]
        if len(filtered_df) > 0:
            channels_urls = url_storage_part + filtered_df['storage_path']
            folder_files(channels_urls, index_name)
            # Prep documents to be uploaded to the vector database (Pinecone)
            loader = DirectoryLoader(f'./{index_name}/', glob="**/*.pdf", loader_cls=PyPDFLoader)
            raw_docs = loader.load()
            # Split documents into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            documents = text_splitter.split_documents(raw_docs)
            logging.info(f"Going to add {len(documents)} chunks to Pinecone")
            # Choose the embedding model and vector store
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            PineconeVectorStore.from_documents(documents=documents, embedding=embeddings, index_name=index_name)
            print("Loading to vectorstore done")


        else:
            print(f"No new files to load for index {index_name}")
            logging.info(f"No new files to load for index {index_name}")
    else:
        print(f"Type {type} not found")
        logging.info(f"Type {type} not found")

##### process for text messages 
def mesage_compiler(df):
    msgs  = {} 
    for index, row in df.iterrows():
        msgs[str(row['users']) + ' '+  str(row['created_at']) ]= row['content']
    return msgs

def load_msg_to_pinecone(index_name, type):
    etl_history = get_rag_supabase(index_name)
    if etl_history is not None:
        start_date = etl_history['updated_at']
    else:
        start_date = None
    
    if type == 'channel':
        channel_messages = pd.DataFrame(supabase.table("channel_messages").select("created_at, channel_id, content , users(username)").execute().data)
        channel_messages['users'] = channel_messages['users'].apply(lambda x :  x['username'] )
        channel_messages=channel_messages[channel_messages['channel_id']==index_name]
        # Convert created_at to UTC instead of localizing
        channel_messages['created_at'] = channel_messages['created_at'].apply(lambda x : pd.to_datetime(x).tz_localize(None))
        if start_date is not None:
            channel_messages = channel_messages[channel_messages['created_at'] > start_date]
        if len(channel_messages) > 0:
            message_channel = mesage_compiler(channel_messages[channel_messages['channel_id']==index_name])
            PineconeVectorStore.from_texts(list(message_channel.values()), OpenAIEmbeddings(), index_name=index_name)
            print("Loading messages done")
            p_index = pc.Index(index_name)
            vectors = []
            for text in list(message_channel.values()):
                # Generate embedding using OpenAI
                response = client.embeddings.create(
                    model="text-embedding-3-small",  # or "text-embedding-3-large" for higher quality
                    input=text
                )
                embedding = response.data[0].embedding
                
                # Create unique ID for the vector
                vector_id = str(uuid.uuid4())
                
                # Create vector tuple with (id, embedding, metadata)
                vector = (vector_id, 
                        embedding,
                        {"text": text})
                
                vectors.append(vector)

            # Upsert vectors to the index in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                p_index.upsert(vectors=batch)
            print("Loading to vectorstore done")



        else:
            print(f"No new messages to load for index {index_name}")
            logging.info(f"No new messages to load for index {index_name}")
    else:
        print(f"Type {type} not found")
        logging.info(f"Type {type} not found")



#### Pinecone functinos 
def create_index(index_name):
    pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws",region="us-east-1") 
                    )
    


def update_index(index_name, type):
    name_index = [x['name'] for x in  pc.list_indexes().indexes]
    if index_name not in name_index:
        create_index(index_name)
        
        load_msg_to_pinecone(index_name, type)
        load_pdf_to_pinecone(index_name, type)

        create_rag_supabase(index_name)
    else:
        print(f"Index {index_name} already exists")
        
        load_msg_to_pinecone(index_name, type)
        load_pdf_to_pinecone(index_name, type)
    return {'resp':'ok'}


###### Generating answers 

def prompt_message(msg):
    promp = f""" 
    You are tasked with answering a message in the tone and style of a specific individual. Below are the details of their tone, language preferences, and example messages.

Message to Respond To:
{msg}

Task:
Generate a response to the message above using the information available in the context and match the tone, vocabulary, and style of the individual as closely as possible
    """
    return promp


def generate_answer(msg, index_name):
    name_index = [x['name'] for x in  pc.list_indexes().indexes]
    if index_name not in name_index:
        return {'resp':'index not found', 'answer':False}
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        document_vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        retriever = document_vectorstore.as_retriever(search_kwargs={"k": 5})
        context = retriever.invoke(msg)
        template = PromptTemplate(template="{query} Context: {context}", input_variables=["query", "context"])
        prompt_with_context = template.invoke({"query": prompt_message(msg), "context": context})
        # Asking the LLM for a response from our prompt with the provided context
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
        results = llm.invoke(prompt_with_context)
        return {'resp':results.content , 'answer':True}