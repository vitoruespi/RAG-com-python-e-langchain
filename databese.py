from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings



folder= 'Base'


def criar_db():
    documentos = carregar_doc()
    chunks = dividir_chunk(documentos)
    vetorizar_chunk(chunks)
  



def carregar_doc():
    carregador = PyPDFDirectoryLoader(folder)#carrega todos os documentos da pasta 
    documentos = carregador.load()#inicia o carregamento
    return documentos

def dividir_chunk(documentos):
    separador = RecursiveCharacterTextSplitter(chunk_size=1200,#tamanho da  chunk em caracteres 
                                             chunk_overlap=125,#quanto que a chunk vai recuar no inicio
                                             length_function=len,#define o tomanho com a função base do python 
                                             add_start_index=True#verifica onde a chunk inicia 
                                             )
    chunks = separador.split_documents(documentos)#divide a lista de documentos passados
    return chunks 

def vetorizar_chunk(chunks):
    pass
