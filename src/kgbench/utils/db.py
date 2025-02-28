import pandas as pd
from lancedb import connect

# Optional[Union[pa.Schema, LanceModel]] = None,
from lancedb.schema import pa


class LanceDB:
    ID_COL: str = "id"
    VECTOR_COL: str = "vector"
    TEXT_COL: str = "text"

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = connect(self.db_path)
        self._required_cols = [
            self.ID_COL, self.TEXT_COL, self.VECTOR_COL]

    def create_table(
            self, table_name: str, schema: pa.Schema=None, overwrite=False):
        """
        Create a new table in the database.

        :param table_name: Name of the table to be created.
        :param schema: Optional. Defines schema for the table.
        """

        if overwrite and table_name in self.conn.table_names():
            self.conn.drop_table(table_name)

        mode = "overwrite" if overwrite else "create"

        self.conn.create_table(
            table_name, schema=schema, mode=mode, exist_ok=True)


    def create_vector_index(
            self, table_name: str, overwrite=False):
        """
        Create an index on a specific table.

        :param table_name: Name of the table to create an index on.
        """
        table = self.conn.open_table(table_name)
        table.create_index(
            vector_column_name=self.VECTOR_COL, replace=overwrite)


    def drop_vector_index(self, table_name: str):
        """
        Delete an existing index from a table.

        :param table_name: Name of the table from which to delete the index.
        """
        table = self.conn.open_table(table_name)
        for i in table.list_indices():
            table.drop_index(str(i))


    def add_documents(self, table_name: str, data: pd.DataFrame):
        """
        Add documents to a specific table.

        :param table_name: Name of the table to which documents will be added.
        :param data: DataFrame containing the document data and vectors.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        # _pd_columns = set([ID_COL, TEXT_COL, VECTOR_COL, "tags", "embed_type"])
        # len(_required_cols - _pd_columns)

        if (set(self._required_cols) - set(data.column.to_list())) > 0:
            raise ValueError(
                f"DataFrame must have required columns: {",".join(self._required_cols)}")

        # Assuming the DataFrame has at least 'id' and 'vector' columns
        table = self.conn.open_table(table_name)
        table.add(data)


    def remove_document(self, table_name: str, document_id):
        """
        Remove a specific document from the table.

        :param table_name: Name of the table from which to remove the document.
        :param document_id: ID of the document to be removed.
        """
        table = self.conn.open_table(table_name)
        table.delete(document_id)

    def update_document(self, table_name: str, document_id, new_data):
        """
        Update a specific document in the table.

        :param table_name: Name of the table where the document is located.
        :param document_id: ID of the document to be updated.
        :param new_data: New data for the document. Must include vector information if vectors are stored.
        """
        # Assuming `new_data` includes a 'vector' field
        table = self.conn.open_table(table_name)
        table.update(document_id, new_data)

    def search(self, table_name: str, query_vector, top_k=10):
        """
        Perform a search on the specified table.

        :param table_name: Name of the table to perform the search on.
        :param query_vector: Vector to be used for querying.
        :param top_k: Number of results to return. Default is 10.
        :return: List of (id, score) tuples representing the most similar documents.
        """
        table = self.conn.open_table(table_name)
        vectors = [query_vector]

        # Fetching and scoring
        scores = table.search(vectors).to_pandas()
        results = [(row['id'], row['score']) for _, row in scores.iterrows()]
        return results


# Example usage:
if __name__ == "__main__":
    db_path = "lancedb_example"
    util = LanceDB(db_path)

    # Create a table with schema
    df_schema = pd.DataFrame(columns=['id', 'text', 'vector'])
    util.create_table("my_table", schema=df_schema)

    # Add some documents
    data = pd.DataFrame([
        {"id": 1, "text": "This is the first document.", "vector": [0.1, 0.2, 0.3]},
        {"id": 2, "text": "This is the second document.", "vector": [0.4, 0.5, 0.6]}
    ])
    util.add_documents("my_table", data)

    # Create an index on the table
    util.create_vector_index("my_table")

    # Search for similar documents
    query_vector = [0.2, 0.3, 0.4]
    results = util.search("my_table", query_vector)
    print(results)

    # Clean up
    util.drop_vector_index("my_table")
    util.remove_document("my_table", 1)
    util.update_document("my_table", 2, {"text": "Updated text", "vector": [0.7, 0.8, 0.9]})
