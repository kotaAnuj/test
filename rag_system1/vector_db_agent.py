
from pinecone import Index, Pinecone, ServerlessSpec
api_key = "b50b0162-43cc-4758-a182-fa7fb1f72eab"
index_name = "quickstart-index"
class VectorDBAgent:
    def __init__(self, index_name, index_host):
        self.api_key = "b50b0162-43cc-4758-a182-fa7fb1f72eab"
        self.index_name = index_name
        self.index_host = index_host
        self.pc = Pinecone(api_key=self.api_key)
        
        # Check if the index already exists
        if self.index_name not in [index.name for index in self.pc.list_indexes().indexes]:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # Replace with your model dimensions
                metric='cosine',  # Replace with your model metric
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        # Retrieve the index host
        index_host = self.pc.describe_index(self.index_name).host

        # Use Index with the host argument
        self.index = Index(name=self.index_name, host=self.index_host,api_key=self.api_key)

       

    def store_embeddings(self, text):
        # Convert data to embeddings and store in Pinecone index
        embeddings = self._get_embeddings(text)
        ids = [str(i) for i in range(len(embeddings))]
        self.index.upsert(vectors=list(zip(ids, embeddings)))

    def search_embeddings(self, query_embedding):
        # Search for similar embeddings in Pinecone index
        results = self.index.query(queries=[query_embedding], top_k=5)
        return results.matches

    def _get_embeddings(self, text):
        # Dummy function to convert data to embeddings
        # Replace with actual implementation
        return [[0.0] * 1536 for _ in text]
