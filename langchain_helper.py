import requests
from bs4 import BeautifulSoup
import re 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from databases.vector_based_DB import FAISS

# SOURCE_URL="https://pulse.zerodha.com/"

VECTOR_DB_PKL_PATH=None
CHUNK_DB_PKL_PATH=None

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n"," "],
    chunk_size=250,
    chunk_overlap=0    
)

def parser(URL,max_load_links=25):
    response = requests.get(URL)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        news_links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if not re.search(r'https?://(www\.)?(twitter|facebook|zerodha)\.com', href, re.IGNORECASE) and \
                not re.search(r'zerodha', href, re.IGNORECASE) and not href[0]=='#':
                news_links.append(href)

        news_links = [link for link in news_links]
        url_loader = UnstructuredURLLoader(urls=news_links[:max_load_links])
        data = url_loader.load()
        
        docs = splitter.split_documents(data)
        return docs
    else:
        print(f"Failed to fetch the page, status code: {response.status_code}")

def split_into_chunks(parsed_articles):
    # take the page content
    chunks = splitter.split_documents(parsed_articles)
    return chunks 

def main_lang(URL,max_loadable_links):
    parsed_articles = parser(URL=URL,max_load_links=max_loadable_links)
    chunks = split_into_chunks(parsed_articles)
    return chunks
    # use the vector DB
    # faiss = FAISS(documents=chunks)
    # return faiss.create_db(store_pkl_file_name=VECTOR_DB_PKL_PATH,store_chunks_path=CHUNK_DB_PKL_PATH)

if __name__=='__main__':
    pass
    # main_lang(SOURCE_URL,max_loadable_links=25)