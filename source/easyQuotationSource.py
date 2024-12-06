# -*- coding: utf-8 -*-
import easyquotation

import inner


class EasyQuotationSource(object):

    def __init__(self, source='sina'):
        self.__quotation = easyquotation.use(source)  # 新浪 ['sina'] 腾讯 ['tencent', 'qq']

    def snapshot(self):
        """判断请求当天是否有数据，并进行解析"""
        resp = self.__quotation.market_snapshot(prefix=True)
        return resp

    def get_stocks_data(self, stocks):
        return self.__quotation.stocks(stocks, prefix=True)


    def day_k_line(self, stocks):
        quotation = inner.use("mydaykline")
        return quotation.real(stocks, prefix=True)

    def time_k_line(self, stocks):
        quotation = easyquotation.use("timekline")
        return quotation.real(stocks, prefix=True)


