from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv





load_dotenv()





Caminho_Db= "db"
prompt_template = """
Responda a pergunta do ususário:
{pergunta}
com base nessa informações: 
{base_conhecimento}
"""



def pergunta():
    pergunta = input("Qual a sua dúvida sobre a nossa biblioteca? \n")


    # carregando o banco de dados


    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")#modelo que vai fazer a comparação com a nossa db 

    db= Chroma(persist_directory=Caminho_Db,
            embedding_function= embedding      
            )

        # Traz os resultados junto com a nota de distância
    resultados_com_nota = db.similarity_search_with_score(pergunta, k=15) 

   #comparando as perguntas 
    resultados = db.similarity_search(pergunta, k=8)     #busca por similaridade gerando uam nota de semelhaça 
    if(len(resultados)==0 ):# verifica se a relevância é o suficiente 
        print("Não encotrei informação relvante ")
        return
    textos_resultado= []
    for resultado in resultados:
        texto = resultado.page_content #pega o conteúdo dos resultados 
        textos_resultado.append(texto)#adiciona na lista
    

    base_conehcimento ="\n\n----\n\n".join(textos_resultado)#une os valores 

 
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt= prompt.invoke({"pergunta": pergunta, "base_conhecimento": base_conehcimento})

    modelo = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    cadeia = modelo | StrOutputParser()
    texto_resposta = cadeia.invoke(prompt)
    print(f"Texto da resposta {texto_resposta}")

pergunta()
""" 
   """