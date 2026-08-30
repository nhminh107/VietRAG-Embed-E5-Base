import os

from dotenv import load_dotenv
from sqlalchemy import Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from database.models import GeneralModel, Base

class SQL_Manager:
    def __init__(self, database_url: str | None = None):
        load_dotenv()
        database_url = database_url or os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is not set.")

        self.engine = create_engine(database_url)
        self.con = Session(self.engine)

    def create_general_model(self) -> None:
        Base.metadata.create_all(self.engine)

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

    def close(self) -> None:
        self.con.close()
        self.engine.dispose()

if __name__ == "__main__":
    sql = SQL_Manager()
    sql.create_general_model()
    print(1)