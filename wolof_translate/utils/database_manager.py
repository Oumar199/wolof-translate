from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd


class TranslationMongoDBManager:
    def __init__(self, uri: str, database: str):

        # recuperate the client
        self.client = MongoClient(uri)

        # recuperate the database
        self.db = self.client.get_database(database)

    def insert_documents(self, documents: list, collection: str = "sentences"):

        # insert documents inside a collection
        results = self.db[collection].insert_many(documents)

        return results

    def insert_document(self, document: dict, collection: str = "sentences"):

        assert not "_id" in document

        # get the id of the last sentence (recuperate the max id and add 1 to it)
        max_id = self.get_max_id(collection)

        # add the new sentences
        document["_id"] = max_id + 1

        results = self.db[collection].insert_one(document)

        return results

    def update_document(
        self,
        id: int,
        document: dict,
        collection: str = "sentences",
        update_collection: str = "updated",
    ):

        # recuperate the document to update
        upd_sent = self.db[collection].find_one({"_id": {"$eq": id}})

        # delete the document
        self.db[collection].update_one(
            {"_id": {"$eq": upd_sent["_id"]}}, {"$set": document}
        )

        # add the sentences to the deleted sentences
        upd_sent["_id"] = len(list(self.db[update_collection].find()))

        results = self.db[update_collection].insert_one(upd_sent)

        return results

    def delete_document(
        self, id: int, collection: str = "sentences", del_collection: str = "deleted"
    ):

        # recuperate the document to delete
        del_sent = self.db[collection].find_one({"_id": {"$eq": id}})

        # delete the sentence
        self.db[collection].delete_one({"_id": {"$eq": del_sent["_id"]}})

        # add the sentences to the deleted sentences
        del_sent["_id"] = len(list(self.db[del_collection].find()))

        results = self.db[del_collection].insert_one(del_sent)

        return results

    def get_max_id(self, collection: str = "sentences"):

        # recuperate the maximum id
        id = list(self.db[collection].find().sort("_id", -1).limit(1))[0]["_id"]

        return id

    def save_data_frames(
        self,
        sentences_path: str,
        deleted_path: str,
        collection: str = "sentences",
        del_collection: str = "deleted",
    ):

        # recuperate the new corpora
        new_corpora = pd.DataFrame(list(self.db[collection].find()))

        # recuperate the deleted sentences as a Data Frame
        deleted_df = pd.DataFrame(list(self.db[del_collection].find()))

        # save the data frames as csv files
        new_corpora.set_index("_id", inplace=True)

        deleted_df.set_index("_id", inplace=True)

        new_corpora.to_csv(sentences_path, index=False)

        deleted_df.to_csv(deleted_path, index=False)

    def load_data_frames(
        self, collection: str = "sentences", del_collection: str = "deleted"
    ):

        # recuperate the new corpora
        new_corpora = pd.DataFrame(list(self.db[collection].find()))

        # recuperate the deleted sentences as a Data Frame
        deleted_df = pd.DataFrame(list(self.db[del_collection].find()))

        return new_corpora, deleted_df
