from sqlalchemy import text, func, update
from sqlalchemy.dialects.mysql import insert

from db import Session
from db.models import StockSnapshot, TradingDay, StockProfit, Datasource


class StockSnapshotMapper(object):
    def __init__(self):
        self.session = Session()

    def query(self, date):
        return self.session.query(StockSnapshot).filter(StockSnapshot.date == date).all()

    def query_range(self, from_date, to_date):
        return self.session.query(StockSnapshot).filter(StockSnapshot.date.between(from_date, to_date)).all()

    def query_greater_than(self, from_date):
        return self.session.query(StockSnapshot).filter(StockSnapshot.date >= from_date).all()

    def insert_or_update_batch(self, batch_item) -> int:
        try:
            insert_stmt = insert(StockSnapshot).values(batch_item)
            update_keys = ['name', 'open', 'close', 'now', 'high', 'low', 'buy', 'sell', 'turnover', 'volume',
                           'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'bid1_volume', 'bid2_volume', 'bid3_volume',
                           'bid4_volume', 'bid5_volume',
                           'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask1_volume', 'ask2_volume', 'ask3_volume',
                           'ask4_volume', 'ask5_volume']
            update_columns = {x.name: x for x in insert_stmt.inserted if x.name in update_keys}
            upsert_stmt = insert_stmt.on_duplicate_key_update(**update_columns)
            result = self.session.execute(upsert_stmt)
            self.session.commit()
            return result.rowcount
        except Exception as e:
            print('Some error happened', e.__repr__())
            self.session.rollback()

    def __del__(self):
        self.session.close()


class TradingDayMapper:

    def __init__(self):
        self.session = Session()

    def insert_or_update_batch(self, batch_item):
        try:
            insert_stmt = text("""
                INSERT IGNORE INTO trading_day (date)
                VALUES (:date)
            """)
            result = self.session.execute(insert_stmt, batch_item)
            self.session.commit()
            return result.rowcount
        except Exception as e:
            print('Some error happened', e.__repr__())
            self.session.rollback()

    def query_max(self):
        """
        query recent trading date
        """
        return self.session.query(func.max(TradingDay.date)).scalar()

    def query_next_days(self, date, offset=6):
        """
        query next several days after specific date, default 6
        """
        return self.session.query(TradingDay).filter(TradingDay.date >= date).order_by(TradingDay.date.asc())\
            .limit(offset).all()


    def __del__(self):
        self.session.close()


class StockProfitMapper:

    def __init__(self):
        self.session = Session()

    def insert_or_update_batch(self, batch_item) -> int:
        try:
            insert_stmt = insert(StockProfit).values(batch_item)
            update_keys = ['yield5']
            update_columns = {x.name: x for x in insert_stmt.inserted if x.name in update_keys}
            upsert_stmt = insert_stmt.on_duplicate_key_update(**update_columns)
            result = self.session.execute(upsert_stmt)
            self.session.commit()
            return result.rowcount
        except Exception as e:
            print('Some error happened', e.__repr__())
            self.session.rollback()

    def __del__(self):
        self.session.close()


class DatasourceMapper(object):
    def __init__(self):
        self.session = Session()

    def update_date(self, code, date) -> int:
        try:
            statement = update(Datasource).where(Datasource.code == code).values(date=date)
            result = self.session.execute(statement)
            self.session.commit()
            return result.rowcount
        except Exception as e:
            print('Some error happened', e.__repr__())
            self.session.rollback()

    def query_date(self, code) -> Datasource:
        return self.session.query(Datasource).filter(Datasource.code == code).one()

    def __del__(self):
        self.session.close()
