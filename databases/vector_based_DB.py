from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

class FAISS():
    def __init__(self,documents=None, model_name="all-mpnet-base-v2"):
        self.documents = documents
        self.embedding_model = model_name
        self.db = None
        
    def create_data_embeddings(self):
        encoder = SentenceTransformer(self.embedding_model)
        return encoder.encode([doc.page_content for doc in self.documents])
    
    def create_embedding(self,text_query):
        encoder = SentenceTransformer(self.embedding_model)
        return encoder.encode(text_query)
        
    def create_db(self,store_pkl_file_name,store_chunks_path):
        # if os.path.exists(store_chunks_path) and os.path.exists(store_pkl_file_name):
            # return
        
        vectors = self.create_data_embeddings()
        dim = vectors.shape[-1]
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(vectors)
        
        # index_buffer = faiss.serialize_index(faiss_index)
        
        # with open(store_pkl_file_name,'wb') as dbfile:
        #     pickle.dump(index_buffer,dbfile)
        
        # with open(store_chunks_path,'wb') as chunksfile:
        #     pickle.dump(self.documents,chunksfile)
            
        # print("\n-----DB created and saved successfully !-----\n")
        return faiss_index
    
    def initialize_db(self, stored_db_path,stored_chunks_path):
        # with open(stored_db_path,'rb') as file:
            # index_buffer = pickle.load(file)
            # self.db = faiss.deserialize_index(index_buffer)
        
        # with open(stored_chunks_path,'rb') as chunksfile:
            # self.documents = pickle.load(chunksfile)
        
        # print("\n-----DB initialized !-----\n")
        pass
        
    def search_db(self,query,top_k=2):
        return self.db.search(query,top_k)

if __name__=='__main__':
    pass