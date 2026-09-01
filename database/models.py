from sqlalchemy import Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

class Base(DeclarativeBase):
    pass

class Data(Base):
    __tablename__ = "data"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    anchor: Mapped[str] = mapped_column(Text, nullable=False)
    positive: Mapped[str | None] = mapped_column(Text, nullable=True)
    hard_negative: Mapped[str | None] = mapped_column(Text, nullable=True)

class GeneralModel(Base):
    __tablename__ = "general"

    data_id: Mapped[str] = mapped_column(Text, primary_key=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    topic: Mapped[str | None] = mapped_column(Text, nullable=True)
    anchor: Mapped[str] = mapped_column(Text, nullable=False)
    positive: Mapped[str | None] = mapped_column(Text, nullable=True)
    hard_negative: Mapped[str | None] = mapped_column(Text, nullable=True)

class GeneralTriplet(Base):
    __tablename__ = "general_triplet"

    data_id: Mapped[str] = mapped_column(Text, primary_key=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    topic: Mapped[str | None] = mapped_column(Text, nullable=True)
    anchor: Mapped[str] = mapped_column(Text, nullable=False)
    positive: Mapped[str | None] = mapped_column(Text, nullable=True)
    hard_negative: Mapped[str | None] = mapped_column(Text, nullable=True)


class LegalModel(Base):
    __tablename__ = "legal"

    data_id: Mapped[str] = mapped_column(Text, primary_key=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    anchor: Mapped[str] = mapped_column(Text, nullable=False)
    positive: Mapped[str | None] = mapped_column(Text, nullable=True)
    hard_negative: Mapped[str | None] = mapped_column(Text, nullable=True)
