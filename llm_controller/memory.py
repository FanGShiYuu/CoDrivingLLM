import random
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import os

api_key = 'your key here'
os.environ["OPENAI_API_KEY"] = api_key

class DrivingMemory:
    def __init__(self, env) -> None:
        self.embedding = OpenAIEmbeddings()
        db_path = './db/' + str(env.spec.id)
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=db_path
        )

        print("==========Loaded ", db_path, " Memory, Now the database has ", len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.==========")


    def retrieveMemory(self, query_scenario, top_k=5):
        """Retrieve the most similar scenarios from memory."""
        similarity_results = self.scenario_memory.similarity_search_with_score(query_scenario, k=top_k)
        fewshot_results = []
        for idx in range(0, len(similarity_results)):
            fewshot_results.append(similarity_results[idx][0].metadata)
        return fewshot_results

    # def retrieveMemory(self, query_scenario, top_k=5):
    #     """Retrieve the most similar scenarios from memory, limited to a certain number of items."""
    #     # Get the first 'limit' embeddings
    #     embeddings_data = self.scenario_memory._collection.get(include=['embeddings', 'documents', 'metadatas'])
    #     limited_embeddings = embeddings_data['embeddings']
    #     limited_documents = embeddings_data['documents']
    #     limited_metadatas = embeddings_data['metadatas']

    #     # Create a temporary Chroma collection with limited items
    #     temp_memory = Chroma(
    #         embedding_function=self.embedding,
    #         persist_directory=None  # Temporary in-memory store
    #     )

    #     # Add the limited data to this temporary collection
    #     temp_memory._collection.add(
    #         embeddings=limited_embeddings,
    #         documents=limited_documents,
    #         metadatas=limited_metadatas
    #     )

    #     # Perform the similarity search on the limited data
    #     similarity_results = temp_memory.similarity_search_with_score(query_scenario, k=top_k)

    #     fewshot_results = []
    #     for idx in range(0, len(similarity_results)):
    #         fewshot_results.append(similarity_results[idx][0].metadata)

    #     return fewshot_results

    def addMemory(self, sce_descrip, human_question, negotiation, action, comments):
        """Add a new scenario to memory."""
        try:
            doc = Document(page_content=sce_descrip, metadata={"human_question": human_question,
                          'negotiation_result': negotiation, 'final_action': action, 'comments': comments})
            self.scenario_memory.add_documents([doc])
            # print(f"Added scenario to memory: {sce_descrip}")
        except Exception as e:
            print(f"Failed to add scenario: {e}")

    def deleteMemory(self, scenario_id):
        """Delete a scenario from memory by its ID."""
        try:
            if scenario_id in self.scenario_memory._collection.ids():
                self.scenario_memory.delete([scenario_id])
                print(f"Deleted scenario with ID: {scenario_id}")
            else:
                print(f"Scenario with ID: {scenario_id} does not exist.")
        except Exception as e:
            print(f"Failed to delete scenario: {e}")

    def combineMemory(self, other_memory):
        """Combine multiple scenarios into a single memory."""
        try:
            other_documents = other_memory.scenario_memory._collection.get(include=['documents', 'metadatas', 'embeddings'])
            current_documents = self.scenario_memory._collection.get(include=['documents', 'metadatas', 'embeddings'])
            for i in range(0, len(other_documents['embeddings'])):
                if other_documents['embeddings'][i] in current_documents['embeddings']:
                    print("Already have one memory item, skip.")
                else:
                    self.scenario_memory._collection.add(
                        embeddings=other_documents['embeddings'][i],
                        metadatas=other_documents['metadatas'][i],
                        documents=other_documents['documents'][i],
                        ids=other_documents['ids'][i]
                    )
            print("Merge complete. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
        except Exception as e:
            print(f"Failed to combine scenarios: {e}")

# Example usage:
# if __name__ == "__main__":
#     dm = DrivingMemory()
#     random_speed = random.randint(10, 50)
#     random_distance = random.randint(10, 500)
#     random_delta_ttcp = 125
#     # Add scenarios
#     dm.addMemory('you are driving at intersection you time to collision is 'f'{random_delta_ttcp} second', 'who pass first', 'other first', 'FASTER', 'bad decision')
#     dm.addMemory('you are driving at roundabout you time to collision is 'f'{random_delta_ttcp} second', 'who pass first', 'ego first', 'FASTER', 'good decision')
#
#     # Retrieve similar scenarios
#     results = dm.retrieveMemory("now you are in a intersection, there is conflict infront of you, your time to colision is 135 second", top_k=2)
#     print("Retrieved scenarios:", results)
#
#     # Combine scenarios
#     # dm.combineMemory(dm)


