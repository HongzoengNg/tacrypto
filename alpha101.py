import pandas as pd
import numpy as np
import datetime


class alpha101(object):
    def __init__(self):
        self.data = None
        self.open_price = None
        self.high_price = None
        self.low_price = None
        self.close_price = None
        self.volume = None

    def input_data(self, data):
        """
        Data is dataframe object with Date, Open, High, Low, Close, Volume, Turnover, VWAP

        """
        self.data = data
        self.open_price = data.Open.values
        self.high_price = data.High.values
        self.low_price = data.Low.values
        self.close_price = data.Close.values
        self.volume = data.Volume.values
        delayed_close = self.delay_series(self.close_price, 1)
        self.returns = self.close_price / delayed_close - 1
        self.vwap = data.VWAP.values
        self.turnover = data.Turnover.values

    def correlation(self, x, y, d):
        return np.corrcoef(x[-d:], y[-d:])[0, 1]

    def correlation_series(self, x, y, d):
        df = pd.DataFrame([x, y]).transpose()
        df.columns = ['x', 'y']
        correlation = df['x'].rolling(d).corr(df['y'])
        return correlation.values

    def covariance(self, x, y, d):

        X = np.stack((x[-d:], y[-d:]), axis=0)
        return np.cov(X)[0, 1]

    def scale(self, x, a=1):
        x_abs = np.abs(x)
        x_sum = np.sum(x_abs)
        return x / x_sum

    def delta(self, x, d):
        return x[-1] - x[-(d + 1)]

    def delta_series(self, x, d):
        return x[d:] - self.delay_series(x, d)[d:]

    def signedpower(self, x, a):
        return np.power(x, a)

    def decay_linear(self, x, d):
        weight = np.arange(d) + 1
        weight = self.scale(weight)
        return np.average(x[-d:], weights=weight)

    def delay(self, x, d):
        return x[-(d + 1)]

    def delay_series(self, x, d):
        return np.roll(x, d)

    def ts_rank(self, x, d):
        ts = x[-d:].copy()
        ranking = ts.argsort().argsort()
        return d - ranking[-1]

    def ts_rank_series(self, x, d):
        length = len(x)
        rank_list = []
        for i in range(length - d):
            j = d + i
            rank = self.ts_rank(x[:j], d)
            rank_list.append(rank)
        return np.array(rank_list)

    def ts_max(self, x, d):
        return np.nanmax(x[-d:])

    def ts_min(self, x, d):
        return np.nanmin(x[-d:])

    def ts_argmin(self, x, d):
        return d - np.argmin(x[-d:])

    def ts_argmax(self, x, d):
        return d - np.argmax(x[-d:])

    def sum_x(self, x, d):
        return np.sum(x[-d:])

    def sum_series(self, x, d):
        x_series = pd.Series(x)
        rolling_sum = x_series.rolling(d).sum()
        return rolling_sum.values

    def adv(self, d):
        return np.mean(self.turnover[-d:])

    # ----------------------------alpha -------------------------------------

    def alpha_006(self, d=10):
        # Alpha#6: (-1 * correlation(open, volume, 10))
        return -1 * self.correlation(self.open_price, self.volume, d)

    def alpha_007(self):
        # Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
        if self.adv(20) < self.volume[-1]:
            return -1 * self.ts_rank(np.abs(self.delta(self.close_price, 7)), 60) * np.sign(
                self.delta(self.close_price, 7))
        else:
            return -1

    def alpha_009(self):
        # Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?
        # delta(close, 1) : (-1 * delta(close, 1))))
        if 0 < self.ts_min(self.delta_series(self.close_price, 1), 5):
            return self.delta(self.close_price, 1)
        else:
            if self.ts_min(self.delta_series(self.close_price, 1), 5) < 0:
                return self.delta(self.close_price, 1)
            else:
                return -1 * self.delta(close, 1)

    def alpha_012(self):
        # Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        return np.sign(self.delta(self.volume, 1)) * (-1 * self.delta(self.close_price, 1))

    def alpha_021(self):
        # Alpha#21: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,
        # 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /
        # adv20) == 1)) ? 1 : (-1 * 1))))
        pass

    def alpha_023(self):
        # Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
        if self.sum_x(self.high_price, 20) / 20 < self.high_price[-1]:
            return -1 * delta(self.high_price, 2)
        else:
            return 0

    def alpha_024(self):
        # Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||
        # ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,
        # 100))) : (-1 * delta(close, 3)))
        a = (self.delta(self.sum_series(self.close_price, 100) / 100, 100) / self.delay(self.close_price, 100)) <= 0.05
        if a:
            return -1 * (self.close_price[-1] - self.ts_min(self.close_price, 100))
        else:
            return -1 * self.delta(self.close_price, 3)

    def alpha_026(self):
        # Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        # print(self.ts_rank_series(self.volume,5),self.ts_rank_series(self.high_price,5))
        a = self.correlation_series(self.ts_rank_series(self.volume, 5), self.ts_rank_series(self.high_price, 5), 5)
        b = self.ts_max(a, 3)
        return -1 * b

    def alpha_028(self):
        # Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
        pass

    def alpha_032(self):
        # Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
        pass

    def alpha_035(self):
        # Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
        a = self.ts_rank(self.volume, 32)
        b = 1 - self.ts_rank(((self.close_price + self.high_price) - self.low_price), 16)
        c = 1 - self.ts_rank(self.returns, 32)
        return a * b * c

    def alpha_041(self):
        # Alpha#41: (((high * low)^0.5) - vwap)
        return (np.power(self.high_price * self.low_price, 0.5) - self.vwap)[-1]

    def alpha_043(self):
        # Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
        return self.ts_rank(self.volume / self.adv(20), 20) * self.ts_rank(-1 * self.delta_series(self.close_price, 7),
                                                                           8)

    def alpha_046(self):
        # Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?
        # (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :
        # ((-1 * 1) * (close - delay(close, 1)))))
        a = (self.delay(self.close_price, 20) - self.delay(self.close_price, 10)) / 10
        b = (self.delay(self.close_price, 10) - self.close_price[-1]) / 10
        if 0.25 < (a - b):
            return -1 * 1
        else:
            if (a - b) < 0:
                return 1
            else:
                return -1 * 1 * (self.close_price[-1] - self.delay(self.close_price, 1))

    def alpha_049(self):
        # Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *
        # 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        a = (self.delay(self.close_price, 20) - self.delay(self.close_price, 10)) / 10
        b = (self.delay(self.close_price, 10) - self.close_price[-1]) / 10
        if (a - b) < -1 * 0.1:
            return 1
        else:
            return -1 * (self.close_price[-1] - self.delay(self.close_price, 1))

    def alpha_051(self):
        # Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *
        # 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        a = (self.delay(self.close_price, 20) - self.delay(self.close_price, 10)) / 10
        b = (self.delay(self.close_price, 10) - self.close_price[-1]) / 10
        if (a - b) < -1 * 0.05:
            return 1
        else:
            return -1 * (self.close_price[-1] - self.delay(self.close_price, 1))

    def alpha_053(self):
        # Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        a = self.close_price - self.low_price
        b = self.high_price - self.close_price
        x = (a - b) / (a)
        return -1 * self.delta(x, 9)

    def alpha_054(self):
        # Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        return (-1 * (self.low_price[-1] - self.close_price[-1]) * (self.open_price[-1] ** 5)) / (
                    (self.low_price[-1] - self.high_price[-1]) * (self.close_price[-1] ** 5))

    def alpha_084(self):
        # Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
        x = self.ts_rank((self.vwap - self.ts_max(self.vwap, 15)), 21)
        a = self.delta(self.close_price, 5)
        return self.signedpower(x, a)

    def alpha_101(self):
        # Alpha#101: ((close - open) / ((high - low) + .001))
        return ((self.close_price[-1] - self.open_price[-1]) / (self.high_price[-1] - self.low_price[-1] + 0.001))


if __name__ == '__main__':
    # -- import data --
    data = pd.read_csv("binance_swap_kline.BTCUSD.csv", index_col=5)
    data.index = [datetime.datetime.fromtimestamp(int(ts) / 1000) for ts in data.index]
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data['Turnover'] = data['Volume'] * data['Close']
    resample_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Turnover': 'sum'
    }
    # -- resample to hourly data --
    data_hour = data.resample('60min').agg(resample_dict)
    data_hour['VWAP'] = data_hour['Turnover'] / data_hour['Volume']

    # -- call alpha 101 --
    al = alpha101()
    al.input_data(data_hour)
    al_101 = al.alpha_101()
    print(al_101)