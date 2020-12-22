import pandas as pd
import numpy as np
import os
import gc
import traceback

from multiprocessing import Pool

TradeDataPath = "./binance_swap_trade/"
MoneyFlowChunkDataPath = "./binance_swap_MoneyFlowFactor/"
if not os.path.isdir(MoneyFlowChunkDataPath):
    os.mkdir(MoneyFlowChunkDataPath)


class TradeToBasicMoneyFlow(object):
    def __init__(self):
        self.th_1, self.th_2, self.th_3 = 5000, 10000, 15000   

        self.activeOrderFuncList = [
            self.getSmallOrderBuy, self.getSmallOrderSell,
            self.getMediumOrderBuy, self.getMediumOrderSell,
            self.getLargeOrderBuy, self.getLargeOrderSell,
            self.getSuperLargeOrderBuy, self.getSuperLargeOrderSell
        ]
    
    # 散户主动做多
    def getSmallOrderBuy(self, df):
        return df[
            (df['turnover'] < self.th_1) & (df['action'] == 'BUY') 
        ][['volume', 'turnover']].sum().rename({'volume': 'SmallBuyOrderVolume', 'turnover': 'SmallBuyOrderTurnover'})

    # 散户主动做空
    def getSmallOrderSell(self, df):
        return df[
            (df['turnover'] < self.th_1) & (df['action'] == 'SELL') 
        ][['volume', 'turnover']].sum().rename({'volume': 'SmallSellOrderVolume', 'turnover': 'SmallSellOrderTurnover'})

    # 中户主动做多
    def getMediumOrderBuy(self, df):
        return df[
            (df['turnover'] >= self.th_1) & (df['turnover'] < self.th_2) & (df['action'] == 'BUY') 
        ][['volume', 'turnover']].sum().rename({'volume': 'MediumBuyOrderVolume', 'turnover': 'MediumBuyOrderTurnover'})

    # 中户主动做空
    def getMediumOrderSell(self, df):
        return df[
            (df['turnover'] >= self.th_1) & (df['turnover'] < self.th_2) & (df['action'] == 'SELL') 
        ][['volume', 'turnover']].sum().rename({'volume': 'MediumSellOrderVolume', 'turnover': 'MediumSellOrderTurnover'})

    # 大户主动做多
    def getLargeOrderBuy(self, df):
        return df[
            (df['turnover'] >= self.th_2) & (df['turnover'] < self.th_3) & (df['action'] == 'BUY') 
        ][['volume', 'turnover']].sum().rename({'volume': 'LargeBuyOrderVolume', 'turnover': 'LargeBuyOrderTurnover'})

    # 大户主动做空
    def getLargeOrderSell(self, df):
        return df[
            (df['turnover'] >= self.th_2) & (df['turnover'] < self.th_3) & (df['action'] == 'SELL') 
        ][['volume', 'turnover']].sum().rename({'volume': 'LargeSellOrderVolume', 'turnover': 'LargeSellOrderTurnover'})

    # 特大户主动做多
    def getSuperLargeOrderBuy(self, df):
        return df[
            (df['turnover'] >= self.th_3) & (df['action'] == 'BUY') 
        ][['volume', 'turnover']].sum().rename({'volume': 'SuperLargeBuyOrderVolume', 'turnover': 'SuperLargeBuyOrderTurnover'})

    # 特大户主动做空
    def getSuperLargeOrderSell(self, df):
        return df[
            (df['turnover'] >= self.th_3) & (df['action'] == 'SELL') 
        ][['volume', 'turnover']].sum().rename({'volume': 'SuperLargeSellOrderVolume', 'turnover': 'SuperLargeSellOrderTurnover'})

    def convert_moneyflow_factor(self, filename):
        try:
            if filename.endswith('csv'):
                datestr = filename.split(".")[-2]
                print("\r" + datestr, end="")
                trade_df = pd.read_csv(TradeDataPath + filename)
                trade_df.columns = ['price', 'volume', 'action', 'time']
                trade_df['time'] = pd.to_datetime(trade_df['time'], unit='ms')
                trade_df.set_index('time', inplace=True)
                trade_df['turnover'] = (trade_df['price'] * trade_df['volume']).round(3)

                MoneyFlow_df_list = []
                for func in self.activeOrderFuncList:
                    tmp_df = trade_df.groupby(pd.Grouper(freq='1min')).apply(func)
                    MoneyFlow_df_list.append(tmp_df)

                MoneyFlow_df = pd.concat(MoneyFlow_df_list, axis=1)

                MoneyFlow_df['NetSuperLargeBuyTurnover'] = MoneyFlow_df['SuperLargeBuyOrderTurnover'] - MoneyFlow_df['SuperLargeSellOrderTurnover']
                MoneyFlow_df['NetLargeBuyTurnover'] = MoneyFlow_df['LargeBuyOrderTurnover'] - MoneyFlow_df['LargeSellOrderTurnover']
                MoneyFlow_df['NetMediumBuyTurnover'] = MoneyFlow_df['MediumBuyOrderTurnover'] - MoneyFlow_df['MediumSellOrderTurnover']
                MoneyFlow_df['NetSmallBuyTurnover'] = MoneyFlow_df['SmallBuyOrderTurnover'] - MoneyFlow_df['SmallSellOrderTurnover']

                MoneyFlow_df['NetSuperLargeBuyVolume'] = MoneyFlow_df['SuperLargeBuyOrderVolume'] - MoneyFlow_df['SuperLargeSellOrderVolume']
                MoneyFlow_df['NetLargeBuyVolume'] = MoneyFlow_df['LargeBuyOrderVolume'] - MoneyFlow_df['LargeSellOrderVolume']
                MoneyFlow_df['NetMediumBuyVolume'] = MoneyFlow_df['MediumBuyOrderVolume'] - MoneyFlow_df['MediumSellOrderVolume']
                MoneyFlow_df['NetSmallBuyVolume'] = MoneyFlow_df['SmallBuyOrderVolume'] - MoneyFlow_df['SmallSellOrderVolume']

                # 净流入金额
                MoneyFlow_df['NetFlowTurnover'] = MoneyFlow_df[['NetSuperLargeBuyTurnover', 'NetLargeBuyTurnover', 'NetMediumBuyTurnover', 'NetSmallBuyTurnover']].sum(axis=1)

                # 净流入量
                MoneyFlow_df['NetFlowVolume'] = MoneyFlow_df[['NetSuperLargeBuyVolume', 'NetLargeBuyVolume', 'NetMediumBuyVolume', 'NetSmallBuyVolume']].sum(axis=1)

                # 总交易量
                MoneyFlow_df['volume'] = MoneyFlow_df[
                    [
                        'SuperLargeBuyOrderVolume', 'SuperLargeSellOrderVolume', 'LargeBuyOrderVolume', 'LargeSellOrderVolume',
                        'MediumBuyOrderVolume', 'MediumSellOrderVolume', 'SmallBuyOrderVolume', 'SmallSellOrderVolume'
                    ]
                ].sum(axis=1)

                # 总交易额
                MoneyFlow_df['turnover'] = MoneyFlow_df[
                    [
                        'SuperLargeBuyOrderTurnover', 'SuperLargeSellOrderTurnover', 'LargeBuyOrderTurnover', 'LargeSellOrderTurnover',
                        'MediumBuyOrderTurnover', 'MediumSellOrderTurnover', 'SmallBuyOrderTurnover', 'SmallSellOrderTurnover'
                    ]
                ].sum(axis=1)

                # 净流入率（金额）
                MoneyFlow_df['NetFlowTurnover_Rate'] = MoneyFlow_df['NetFlowTurnover'] / MoneyFlow_df['turnover']

                # 大单净流入量
                MoneyFlow_df['LargeOrderVolume'] = MoneyFlow_df['NetSuperLargeBuyVolume'] + MoneyFlow_df['NetLargeBuyVolume']
                # 大单净流入金额
                MoneyFlow_df['LargeOrderTurnover'] = MoneyFlow_df['NetSuperLargeBuyTurnover'] + MoneyFlow_df['NetLargeBuyTurnover']
                # 大单流入率(金额)
                MoneyFlow_df['NetFlowLargeOrderTurnover_Rate'] = MoneyFlow_df['LargeOrderTurnover'] / MoneyFlow_df['turnover']

                # 小单净流入量
                MoneyFlow_df['SmallOrderVolume'] = MoneyFlow_df['NetSmallBuyVolume'] + MoneyFlow_df['NetMediumBuyVolume']
                # 小单净流入金额
                MoneyFlow_df['SmallOrderTurnover'] = MoneyFlow_df['NetSmallBuyTurnover'] + MoneyFlow_df['NetMediumBuyTurnover']
                # 小单流入率(金额)
                MoneyFlow_df['NetFlowSmallOrderTurnover_Rate'] = MoneyFlow_df['SmallOrderTurnover'] / MoneyFlow_df['turnover']

                # 主动做多量
                MoneyFlow_df['BuyOrderVolume'] = MoneyFlow_df[['SmallBuyOrderVolume', 'MediumBuyOrderVolume', 'LargeBuyOrderVolume', 'SuperLargeBuyOrderVolume']].sum(axis=1)
                # 主动做空量
                MoneyFlow_df['SellOrderVolume'] = MoneyFlow_df[['SmallSellOrderVolume', 'MediumSellOrderVolume', 'LargeSellOrderVolume', 'SuperLargeSellOrderVolume']].sum(axis=1)

                # 合约多空比
                MoneyFlow_df['LongShortRatio'] = MoneyFlow_df['BuyOrderVolume'] / MoneyFlow_df['SellOrderVolume']

                MoneyFlow_df.round(3).to_csv(MoneyFlowChunkDataPath + "{}.csv".format(datestr))
                
                del trade_df
                gc.collect()
        except:
            print(traceback.format_exc())

    def parallel_convert(self):
        p = Pool(2)
        for filename in os.listdir(TradeDataPath)[:5]:
            p.apply_async(self.convert_moneyflow_factor, (filename,))
        p.close()
        p.join()
    

class MoneyFactorReader(object):
    def __init__(self):
        pass

    def getMoneyFactor(self, save_to_file=False):
        money_flow_df_list = []
        for filename in os.listdir(MoneyFlowChunkDataPath):
            if filename.endswith('.csv'):
                df = pd.read_csv(MoneyFlowChunkDataPath + filename, index_col=0)
                money_flow_df_list.append(df)
        
        money_flow_df = pd.concat(money_flow_df_list)
        money_flow_df = money_flow_df.groupby(money_flow_df.index).sum()

        # 主动做多短长期移动平均差
        money_flow_df['LS_TotalBuyOrderVolume'] = money_flow_df['BuyOrderVolume'].rolling(2).mean().fillna(0) - money_flow_df['BuyOrderVolume'].rolling(3).mean().fillna(0)

        # 主动做空短长期移动平均差
        money_flow_df['LS_TotalSellOrderVolume'] = money_flow_df['SellOrderVolume'].rolling(2).mean().fillna(0) - money_flow_df['SellOrderVolume'].rolling(3).mean().fillna(0)

        # 主动做多短长期移动平均差 - 主动做空短长期移动平均差
        money_flow_df['LS_DeltaBuySellVolume'] = money_flow_df['LS_TotalBuyOrderVolume'] - money_flow_df['LS_TotalSellOrderVolume']

        # 大单主动做多短长期移动平均差
        x1 = money_flow_df['SuperLargeBuyOrderVolume'] + money_flow_df['LargeBuyOrderVolume']
        money_flow_df['LS_LargerBuyOrderVolume'] = x1.rolling(2).mean().fillna(0) - x1.rolling(3).mean().fillna(0)

        # 大单主动做空短长期移动平均差
        x2 = money_flow_df['SuperLargeSellOrderVolume'] + money_flow_df['LargeSellOrderVolume']
        money_flow_df['LS_LargerSellOrderVolume'] = x2.rolling(2).mean().fillna(0) - x2.rolling(3).mean().fillna(0)

        # 大单主动做多短长期移动平均差 - 大单主动做空短长期移动平均差
        money_flow_df['LS_Delta_LargeBuySellVolume'] = money_flow_df['LS_LargerBuyOrderVolume'] - money_flow_df['LS_LargerSellOrderVolume']

        # 小单主动做多短长期移动平均差
        x1 = money_flow_df['MediumBuyOrderVolume'] + money_flow_df['SmallOrderVolume']
        money_flow_df['LS_SmallBuyOrderVolume'] = x1.rolling(2).mean().fillna(0) - x1.rolling(3).mean().fillna(0)

        # 小单主动做空短长期移动平均差
        x2 = money_flow_df['MediumSellOrderVolume'] + money_flow_df['SmallSellOrderVolume']
        money_flow_df['LS_SmallSellOrderVolume'] = x2.rolling(2).mean().fillna(0) - x2.rolling(3).mean().fillna(0)

        # 小单主动做多短长期移动平均差 - 小单主动做空短长期移动平均差
        money_flow_df['LS_Delta_SmallBuySellVolume'] = money_flow_df['LS_SmallBuyOrderVolume'] - money_flow_df['LS_SmallSellOrderVolume']

        # 大小单总体平均差方向
        money_flow_df['LS_Delta_Direction'] = money_flow_df['LS_Delta_LargeBuySellVolume'] + money_flow_df['LS_Delta_SmallBuySellVolume']

        self.moneyflow_factors = money_flow_df.round(3)

        if save_to_file:
            self.moneyflow_factors.to_csv('./moneyflow.csv')

        return self.moneyflow_factors



# if __name__ == "__main__":
    # basic_mf = TradeToBasicMoneyFlow()
    # basic_mf.parallel_convert()
    # mfr = MoneyFactorReader()
    # mfr.getMoneyFactor(save_to_file=False)

