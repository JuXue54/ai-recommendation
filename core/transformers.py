import datetime
from datetime import timedelta

from db.mappers import StockSnapshotMapper, TradingDayMapper, StockProfitMapper, DatasourceMapper


class FinancialDataTransformer:

    def sync_trading_day(self, date=None):
        trading_day_mapper = TradingDayMapper()
        if date is None:
            date = trading_day_mapper.query_max()
        all = StockSnapshotMapper().query_greater_than(date)
        dates = set()
        for x in all:
            dates.add(x.date)
        res = list(map(lambda d: {'date': d}, sorted(dates)))
        return trading_day_mapper.insert_or_update_batch(res)

    def sync_future_5_days_yield(self, date) -> int:
        """
        compute yield of all stocks in future 5 days
        """
        from_date = date
        dates = TradingDayMapper().query_next_days(from_date)
        to_date = dates[-1]
        if len(dates) < 6:
            print(f"Not have enough dates, dates: {dates}")
            return 0
        print(f'Query close price from {from_date} to {to_date}')
        all = StockSnapshotMapper().query_range(from_date, to_date)
        m = {}
        for x in all:
            sub_map = m.get(x.code, {})
            sub_map[x.date] = x.close
            m[x.code] = sub_map

        res = []
        for k, v in m.items():
            lis = list(map(lambda a: a[1], sorted(v.items())))
            try:
                returns = (lis[-1] - lis[0]) / lis[0]
            except ZeroDivisionError:
                returns = 0
            res.append({"yield5": returns, "code": k, "date": date})

        # print(res)
        rows = StockProfitMapper().insert_or_update_batch(res)
        return rows

    def sync_all_future_yield(self, curr=datetime.date.today()):
        """
        compute yield of all stocks in future 5 days
        """
        datasource_mapper = DatasourceMapper()
        while True:
            datasource = datasource_mapper.query_date('FUTURE_5_YIELD')
            date = datasource.date
            if date >= curr:
                print(f"current datasource is {date}")
                break
            dates = TradingDayMapper().query_next_days(date, 2)
            if len(dates) < 2:
                print(f"No enough dates here, current date is {date}")
                break
            rows = self.sync_future_5_days_yield(dates[1])
            if rows == 0:
                print(f"Since affected rows are 0, Future yield failed to update! Current datasource is {date}")
                break
            datasource_mapper.update_date('FUTURE_5_YIELD', dates[1])
            print(f"Affected rows are {rows}, current datasource is {dates[1]}")
