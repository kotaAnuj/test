from neo4j import GraphDatabase

class GraphDBAgent:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def store_data(self, data):
        with self.driver.session() as session:
            session.write_transaction(self._create_and_return_node, data)

    @staticmethod
    def _create_and_return_node(tx, data):
        query = "CREATE (a:Data {info: $data}) RETURN a"
        result = tx.run(query, data=data)
        return result.single()[0]
