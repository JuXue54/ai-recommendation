from easyquotation.daykline import DayKline


class MyDayKline(DayKline):

    def _gen_stock_prefix(self, stock_codes, day=1500):
        return ["sh{},day,,,{},qfq".format(code, day) for code in stock_codes]