from pydantic import BaseModel

class GeneralModel(BaseModel):
    data_id: str
    source: str | None = None
    title: str | None = None
    topic: str | None = None
    anchor: str
    positive: str| None = None
    hard_negative: str | None = None