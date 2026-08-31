import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import GeneralModel, LegalModel

class SQL_Manager:
    def __init__(self, database_url: str | None = None):
        load_dotenv()
        database_url = database_url or os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is not set.")

        self.engine = create_engine(database_url)
        self.con = Session(self.engine)

    def create_general_model(self) -> None:
        GeneralModel.__table__.create(self.engine, checkfirst=True)

    def insert_general_model(self, data: GeneralModel) -> None:
        self.con.add(
            GeneralModel(
                data_id=data.data_id,
                source=data.source,
                title=data.title,
                topic=data.topic,
                anchor=data.anchor,
                positive=data.positive,
                hard_negative=data.hard_negative,
            )
        )

    def create_legal_model(self) -> None:
        LegalModel.__table__.create(self.engine, checkfirst=True)

    def insert_legal_model(self, data: LegalModel) -> None:
        self.con.add(
            LegalModel(
                data_id=data.data_id,
                source=data.source,
                title=data.title,
                anchor=data.anchor,
                positive=data.positive,
                hard_negative=data.hard_negative,
            )
        )

    def close(self) -> None:
        self.con.close()
        self.engine.dispose()

if __name__ == "__main__":
    sql = SQL_Manager()
    try:
        #sql.create_general_model()
        sql.create_legal_model()
    finally:
        sql.close()
