import os
import pprint

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.llms import Ollama
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import MergerRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.embeddings import SentenceTransformerEmbeddings

groq_api_key=os.environ['GROQ_API_KEY']
load_dotenv()

def get_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2",encode_kwargs = {'normalize_embeddings': False} ) 
    return embeddings
    # huggingface_embeddings = HuggingFaceBgeEmbeddings(
    #         model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    #         #model_name="jinaai/jina-embeddings-v2-base-es",
    #         model_kwargs={'device':'cpu', 'trust_remote_code': True}, 
    #         encode_kwargs={'normalize_embeddings': False, 'attn_implementation': "eager"},
    #     )
    # return huggingface_embeddings

def get_db():
    from langchain_postgres.vectorstores import PGVector

    connection_string = "postgresql://postgres:postgres@localhost:5432/db_subvenciones"

    db = PGVector(
        collection_name="subvenciones",
        connection=connection_string,
        embeddings=get_embeddings(),
    )
    return db
    
def get_llm():
    #retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    llm=ChatGroq(groq_api_key=groq_api_key,
    model_name="llama3-8b-8192")    
    return llm

def get_selfQueryRetriever():
    from langchain.chains.query_constructor.base import AttributeInfo

    metadata_field_info = [
        AttributeInfo(
            name="Destinatarios",
            description="Empresas, organizaciones o personas que reciben la ayuda",
            type="string",
        ),
        AttributeInfo(
            name="Organismo",
            description="Organismo que da la ayuda (Ayuntamiento, Consejería, etc.)",
            type="string",
        ),
        AttributeInfo(
            name="Referencia",
            description="Número de referencia de la convocatoria (breve descripción)",
            type="string",
        ),
        AttributeInfo(
            name="Título",
            description="Título de la convocatoria (breve descripción)",
            type="string",
        ),
        AttributeInfo(
            name="Sector", 
            description="Sector al que se destina la convocatoria ", 
            type="string"
        ),
        AttributeInfo(
            name="Subsector", 
            description="Subsector dentro del Sector, completa la información del atributo Sector.", 
            type="string"
        ),
        AttributeInfo(
            name="Tipo", 
            description="Tipo de ayudas", 
            type="string"
        ),
        AttributeInfo(
            name="AmbitoGeografico", 
            description="Define en qué territorio (ciudad, comunidad o región), se concede la ayuda.", 
            type="string"
        ),
        AttributeInfo(
            name="Información Adicional", 
            description="Otra información relevante al documento.", 
            type="string"
        ),
        AttributeInfo(
            name="Referencias de la Publicación", 
            description="Otros documentos relacionados con este documento.", 
            type="string"
        ),
        AttributeInfo(
            name="Plazo de solicitud", 
            description="Plazo de solicitud de la ayuda, subvención o premio.", 
            type="string"
        ),
        
        ]
        
    selfqueryRetriever = SelfQueryRetriever.from_llm(
        get_llm(),
        get_db(),
        "Subvenciones y ayudas",
        metadata_field_info,
        search_kwargs={'k': 15}
        )   
    return selfqueryRetriever

    
def get_retriever():
    print("get_retriever")
    retriever = get_db().as_retriever(search_type="similarity", search_kwargs={"k": 15})
    # docs = retriever.get_relevant_documents("vehículos eléctricos")
    # pprint.pp(docs)
    return retriever
    
    
def get_mergeRetriever():    
    print("get_mergeRetriever")
    mergeRetriever = MergerRetriever(retrievers=[get_selfQueryRetriever(), get_retriever()])
    return mergeRetriever


def get_compressionRetriever():
    model = HuggingFaceCrossEncoder(model_name="mixedbread-ai/mxbai-rerank-xsmall-v1",
                                    model_kwargs={'trust_remote_code': True})
    compressor = CrossEncoderReranker(model=model, top_n=10)
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=get_mergeRetriever()
    )
    print("get_compressionRetriever")
    return compression_retriever
    
def get_response(question):
    if question == "": return   
    qa_template = """Eres un asistente para responder preguntas basándote en los documentos proporcionados.
                    Proporciona la respuesta a la pregunta basándote en la información encontrada en los documentos. 
                    Resume brevemente la información relevante para la respuesta.
                    Si no encuentras la respuesta en los documentos, indica claramente que no se encontró la información.

{context}
        "\n\n"
        "{context}"

    Pregunta: {question}
    Respuesta:"""
  
    prompt = PromptTemplate(template=qa_template,
                                input_variables=['context','question'])
    combine_custom_prompt='''
        Eres un asistente para combinar las respuestas encontradas en múltiples documentos.
        Resume las respuestas encontradas en los documentos. 
        Si un documento no contiene información relevante, omítelo del resumen.
        Ordena la información de manera que sea clara y concisa.

    Text:`{context}`
    '''
    """ 
    """ 
    combine_prompt_template = PromptTemplate(
        template=combine_custom_prompt, 
        input_variables=['context']
    )
    qa_chain = RetrievalQA.from_chain_type(get_llm(), 
                                           retriever=get_compressionRetriever(), 
                                           chain_type="map_reduce",
                                           return_source_documents=True,
    chain_type_kwargs= {
            "verbose": False,
            "question_prompt": prompt,
            "combine_prompt": combine_prompt_template,
            "combine_document_variable_name": "context"})

    """ qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=compression_retriever,
                                            return_source_documents=True) """


    #query = """ ¿Quiénes son los beneficiarios de la referencia 71572 ?   """  
    #question = query

    result=qa_chain.invoke(question)
    return result

    
# question = """ ¿Qué subvenciones se dan en el Ayuntamiento de Basauri?   """  
# result = get_response(question)
# pprint.pp(result)

