from app.cypher.value_mapper import ValueMapper


class Mapper:
    def __init__(self):
        self.value_mapper = ValueMapper()

    def map(self, query: str):
        results = self.value_mapper.map_query(query)

        return results
