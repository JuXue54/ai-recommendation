from sqlalchemy import Column, BIGINT, String, Date, DATE, DOUBLE, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()
metadata = Base.metadata


class StockSnapshot(Base):
    __tablename__ = 'stock_snapshot'
    id = Column(BIGINT, primary_key=True, autoincrement=True)
    code = Column(String)
    date = Column(DATE)
    name = Column(String)
    open = Column(DOUBLE)
    close = Column(DOUBLE)
    now = Column(DOUBLE)
    high = Column(DOUBLE)
    low = Column(DOUBLE)
    buy = Column(DOUBLE)
    sell = Column(DOUBLE)
    turnover = Column(DOUBLE)
    volume = Column(DOUBLE)
    bid1 = Column(DOUBLE)
    bid1_volume = Column(DOUBLE)
    bid2 = Column(DOUBLE)
    bid2_volume = Column(DOUBLE)
    bid3 = Column(DOUBLE)
    bid3_volume = Column(DOUBLE)
    bid4 = Column(DOUBLE)
    bid4_volume = Column(DOUBLE)
    bid5 = Column(DOUBLE)
    bid5_volume = Column(DOUBLE)
    ask1 = Column(DOUBLE)
    ask1_volume = Column(DOUBLE)
    ask2 = Column(DOUBLE)
    ask2_volume = Column(DOUBLE)
    ask3 = Column(DOUBLE)
    ask3_volume = Column(DOUBLE)
    ask4 = Column(DOUBLE)
    ask4_volume = Column(DOUBLE)
    ask5 = Column(DOUBLE)
    ask5_volume = Column(DOUBLE)

    __table_args__ = (UniqueConstraint("code", "date", name='uqk_code_date'),)


class StockProfit(Base):
    __tablename__ = 'stock_profit'

    id = Column(BIGINT, primary_key=True, autoincrement=True)
    code = Column(String)
    date = Column(DATE)
    yield5 = Column(DOUBLE, comment='5 day yields')

    __table_args__ = (UniqueConstraint("code", "date", name='uqk_code_date'),)


class TradingDay(Base):
    __tablename__ = 'trading_day'

    date = Column(DATE, primary_key=True)

class Datasource(Base):
    __tablename__ = 'datasource'

    id = Column(BIGINT, primary_key=True, autoincrement=True)
    code = Column(String)
    name = Column(String)
    date = Column(DATE)
    __table_args__ = (UniqueConstraint("code", name='uniq_code'),)
