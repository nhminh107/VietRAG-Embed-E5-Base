from pydantic import BaseModel

class GeneralModel(BaseModel):
    data_id: str
    source: str
    title: str
    topic: str
    anchor: str
    positive: str
    hard_negative: str