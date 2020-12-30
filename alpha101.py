import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class alpha101(object):
    def __init__(self):
        self.data = None
        self.open_price = None
        self.high_price = None
        self.low_price = None
        self.close_price = None
        self.volume = None
        self.alpha_list = []

        self.initialization()
        self.default_params = {}
        self.default_params['alpha_006'] = {'d_1': 10}
        self.default_params['alpha_007'] = {'d_1': 20, 'd_2': 7, 'd_3': 60, 'd_4': 7, 'c_1': 1}
        self.default_params['alpha_009'] = {'c_1': 0, 'd_1': 1, 'd_2': 5, 'd_3': 1}
        self.default_params['alpha_012'] = {'d_1': 1, 'd_2': 1}
        self.default_params['alpha_023'] = {'d_1': 20, 'c_1': 20, 'd_2': 2, 'c_2': 0}
        self.default_params['alpha_024'] = {'d_1': 100, 'c_1': 0.05, 'd_2': 3}
        self.default_params['alpha_026'] = {'d_1': 5, 'd_2': 3}
        self.default_params['alpha_035'] = {'d_1': 32, 'd_2': 16, 'd_3': 32}
        self.default_params['alpha_041'] = {'c_1': 0.5}
        self.default_params['alpha_043'] = {'d_1': 20, 'd_2': 7, 'd_3': 8}
        self.default_params['alpha_046'] = {'d_1': 20, 'd_2': 10, 'd_3': 10, 'c_1': 0.25, 'c_2': 1, 'c_3': 0, 'c_4': 1,
                                            'c_5': 1}
        self.default_params['alpha_049'] = {'d_1': 20, 'd_2': 10, 'd_3': 10, 'c_1': 0.1, 'c_2': 1, 'd_4': 1}
        self.default_params['alpha_051'] = {'d_1': 20, 'd_2': 10, 'd_3': 10, 'c_1': 0.05, 'c_2': 1, 'd_4': 1}
        self.default_params['alpha_053'] = {'d_1': 9}
        self.default_params['alpha_054'] = {'c_1': 5}
        self.default_params['alpha_084'] = {'d_1': 15, 'd_2': 21, 'd_3': 5}
        self.default_params['alpha_101'] = {'c_1': 0.001}

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

    def alpha_006(self, params=None):
        # Alpha#6: (-1 * correlation(open, volume, 10))
        if params == None:
            params = self.default_params['alpha_006']
        d_1 = params['d_1']
        return -1 * self.correlation(self.open_price, self.volume, d_1)

    def alpha_007(self, params=None):
        # Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
        if params == None:
            params = self.default_params['alpha_007']
        d_1 = params['d_1']
        d_2 = params['d_2']
        d_3 = params['d_3']
        d_4 = params['d_4']
        c_1 = params['c_1']

        if self.adv(d_1) < self.volume[-1]:
            return -1 * self.ts_rank(np.abs(self.delta(self.close_price, d_2)), d_3) * np.sign(
                self.delta(self.close_price, d_4))
        else:
            return -1 * c_1

    def alpha_009(self, params=None):
        # Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?
        # delta(close, 1) : (-1 * delta(close, 1))))
        if params == None:
            params = self.default_params['alpha_009']
        c_1 = params['c_1']
        d_1 = params['d_1']
        d_2 = params['d_2']
        d_3 = params['d_3']

        if c_1 < self.ts_min(self.delta_series(self.close_price, d_1), d_2):
            return self.delta(self.close_price, d_1)
        else:
            if self.ts_min(self.delta_series(self.close_price, d_1), d_2) < c_1:
                return self.delta(self.close_price, d_1)
            else:
                return -1 * self.delta(self.close_price, d_3)

    def alpha_012(self, params=None):
        # Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        if params == None:
            params = self.default_params['alpha_012']
        d_1 = params['d_1']
        d_2 = params['d_2']
        return np.sign(self.delta(self.volume, d_1)) * (-1 * self.delta(self.close_price, d_2))

    #     def alpha_021(self):
    #         # Alpha#21: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,
    #         # 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /
    #         # adv20) == 1)) ? 1 : (-1 * 1))))
    #         pass
    def alpha_023(self, params=None):
        # Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
        if params == None:
            params = self.default_params['alpha_023']
        d_1 = params['d_1']
        d_2 = params['d_2']
        c_1 = params['c_1']
        c_2 = params['c_2']

        if self.sum_x(self.high_price, d_1) / c_1 < self.high_price[-1]:
            return -1 * self.delta(self.high_price, d_2)
        else:
            return c_2

    def alpha_024(self, params=None):
        # Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||
        # ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,
        # 100))) : (-1 * delta(close, 3)))
        if params == None:
            params = self.default_params['alpha_024']
        d_1 = params['d_1']
        d_2 = params['d_2']
        c_1 = params['c_1']
        a = (self.delta(self.sum_series(self.close_price, d_1) / d_1, d_1) / self.delay(self.close_price, d_1)) <= c_1
        if a:
            return -1 * (self.close_price[-1] - self.ts_min(self.close_price, d_1))
        else:
            return -1 * self.delta(self.close_price, d_2)

    def alpha_026(self, params=None):
        # Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        # print(self.ts_rank_series(self.volume,5),self.ts_rank_series(self.high_price,5))
        if params == None:
            params = self.default_params['alpha_026']
        d_1 = params['d_1']
        d_2 = params['d_2']
        a = self.correlation_series(self.ts_rank_series(self.volume, d_1), self.ts_rank_series(self.high_price, d_1),
                                    d_1)
        b = self.ts_max(a, d_2)
        return -1 * b

    #     def alpha_028(self):
    #         # Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    #         pass

    #     def alpha_032(self):
    #         # Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
    #         pass

    def alpha_035(self, params=None):
        # Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
        if params == None:
            params = self.default_params['alpha_035']
        d_1 = params['d_1']
        d_2 = params['d_2']
        d_3 = params['d_3']
        a = self.ts_rank(self.volume, d_1)
        b = 1 - self.ts_rank(((self.close_price + self.high_price) - self.low_price), d_2)
        c = 1 - self.ts_rank(self.returns, d_3)
        return a * b * c

    def alpha_041(self, params=None):
        # Alpha#41: (((high * low)^0.5) - vwap)
        if params == None:
            params = self.default_params['alpha_041']
        c_1 = params['c_1']
        return (np.power(self.high_price * self.low_price, c_1) - self.vwap)[-1]

    def alpha_043(self, params=None):
        # Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
        if params == None:
            params = self.default_params['alpha_043']
        d_1 = params['d_1']
        d_2 = params['d_2']
        d_3 = params['d_3']
        return self.ts_rank(self.volume / self.adv(d_1), d_1) * self.ts_rank(
            -1 * self.delta_series(self.close_price, d_2), d_3)

    def alpha_046(self, params=None):
        # Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?
        # (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :
        # ((-1 * 1) * (close - delay(close, 1)))))
        if params == None:
            params = self.default_params['alpha_046']
        d_1 = params['d_1']
        d_2 = params['d_2']
        d_3 = params['d_3']
        c_1 = params['c_1']
        c_2 = params['c_2']
        c_3 = params['c_3']
        c_4 = params['c_4']
        c_5 = params['c_5']
        a = (self.delay(self.close_price, d_1) - self.delay(self.close_price, d_2)) / d_2
        b = (self.delay(self.close_price, d_3) - self.close_price[-1]) / d_3
        if c_1 < (a - b):
            return -1 * c_2
        else:
            if (a - b) < c_3:
                return c_4
            else:
                return -1 * c_5 * (self.close_price[-1] - self.delay(self.close_price, 1))

    def alpha_049(self, params=None):
        # Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *
        # 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        if params == None:
            params = self.default_params['alpha_049']
        d_1 = params['d_1']
        d_2 = params['d_2']
        d_3 = params['d_3']
        d_4 = params['d_4']
        c_1 = params['c_1']
        c_2 = params['c_2']
        a = (self.delay(self.close_price, d_1) - self.delay(self.close_price, d_2)) / d_2
        b = (self.delay(self.close_price, d_3) - self.close_price[-1]) / d_3
        if (a - b) < -1 * c_1:
            return c_2
        else:
            return -1 * (self.close_price[-1] - self.delay(self.close_price, d_4))

    def alpha_051(self, params=None):
        # Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *
        # 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        if params == None:
            params = self.default_params['alpha_051']
        d_1 = params['d_1']
        d_2 = params['d_2']
        d_3 = params['d_3']
        d_4 = params['d_4']
        c_1 = params['c_1']
        c_2 = params['c_2']
        a = (self.delay(self.close_price, d_1) - self.delay(self.close_price, d_2)) / d_2
        b = (self.delay(self.close_price, d_3) - self.close_price[-1]) / d_3
        if (a - b) < -1 * c_1:
            return c_2
        else:
            return -1 * (self.close_price[-1] - self.delay(self.close_price, d_4))

    def alpha_053(self, params=None):
        # Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        if params == None:
            params = self.default_params['alpha_053']
        d_1 = params['d_1']
        a = self.close_price - self.low_price
        b = self.high_price - self.close_price
        x = (a - b) / (a)
        return -1 * self.delta(x, d_1)

    def alpha_054(self, params=None):
        # Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        if params == None:
            params = self.default_params['alpha_054']
        c_1 = params['c_1']
        return (-1 * (self.low_price[-1] - self.close_price[-1]) * (self.open_price[-1] ** c_1)) / (
                    (self.low_price[-1] - self.high_price[-1]) * (self.close_price[-1] ** c_1))

    def alpha_084(self, params=None):
        # Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
        if params == None:
            params = self.default_params['alpha_084']
        d_1 = params['d_1']
        d_2 = params['d_2']
        d_3 = params['d_3']
        x = self.ts_rank((self.vwap - self.ts_max(self.vwap, d_1)), d_2)
        a = self.delta(self.close_price, d_3)
        return self.signedpower(x, a)

    def alpha_101(self, params=None):
        # Alpha#101: ((close - open) / ((high - low) + .001))
        if params == None:
            params = self.default_params['alpha_101']
        c_1 = params['c_1']
        return ((self.close_price[-1] - self.open_price[-1]) / (self.high_price[-1] - self.low_price[-1] + c_1))

    # ------------ parameters optimisation and analysis -----------------

    def initialization(self):
        internal_methods = dir(self)
        alpha_list = []
        for method in internal_methods:
            if 'alpha' in method and 'list' not in method:
                alpha_list.append(method)
        self.alpha_list = alpha_list

    def compute_all(self):
        alpha_values = []
        for alpha in self.alpha_list:
            func = getattr(self, alpha)
            val = func()
            alpha_values.append(val)
        return alpha_values


class FactorsAnalyzer(object):
    def __init__(self, alpha101):
        self.alphaModel = alpha101()
        self.data = None
        self.resampled_data = None
        self.alpha_list = None
        self.resample_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Turnover': 'sum'
        }
        self.alpha_data = None

    def add_data(self, data, freq='60min'):
        self.data = data
        self.resampling(freq)

    def resampling(self, freq='60min'):
        data_hour = self.data.resample(freq).agg(self.resample_dict)
        data_hour['VWAP'] = data_hour['Turnover'] / data_hour['Volume']
        self.resampled_data = data_hour

    def initialization(self):
        self.alphaModel.input_data(self.resampled_data)

        alpha_data = {}
        data_length = len(self.resampled_data.index)
        for i in range(data_length - 120):
            slice_data = self.resampled_data.iloc[i:i + 121]
            date = slice_data.index[-1]
            self.alphaModel.input_data(slice_data)
            alpha_val = self.alphaModel.compute_all()
            alpha_data[date] = alpha_val
        alpha_df = pd.DataFrame.from_dict(alpha_data, orient='index')
        alpha_df.columns = self.alphaModel.alpha_list
        self.alpha_list = self.alphaModel.alpha_list
        data_analysis = self.resampled_data
        data_analysis['t_1'] = data_analysis['Close'].shift(-1) / data_analysis['Close'] - 1
        data_analysis['t_2'] = data_analysis['Close'].shift(-2) / data_analysis['Close'] - 1
        data_analysis['t_3'] = data_analysis['Close'].shift(-3) / data_analysis['Close'] - 1
        data_analysis['t_4'] = data_analysis['Close'].shift(-4) / data_analysis['Close'] - 1
        data_analysis['t_5'] = data_analysis['Close'].shift(-5) / data_analysis['Close'] - 1
        data_analysis['t_6'] = data_analysis['Close'].shift(-6) / data_analysis['Close'] - 1
        df = data_analysis.join(alpha_df)
        self.alpha_data = df

    def analysis(self):
        columns_list = ['t_1', 't_2', 't_3', 't_4', 't_5', 't_6']
        for alpha in self.alpha_list:
            alpha_name = [alpha]
            columns_selected = alpha_name + columns_list
            df_alpha = df[columns_selected]
            df_alpha.dropna(inplace=True)

            total_sample = len(df_alpha.index)
            in_sample_len = round(total_sample * 0.6)
            df_alpha_in_sample = df_alpha.iloc[:in_sample_len]
            df_alpha_out_sample = df_alpha.iloc[in_sample_len:]

            self.in_samples_plot(df_alpha_in_sample[alpha], df_alpha_in_sample[columns_list])
            mean_value = np.mean(df_alpha_in_sample[alpha])

            self.out_samples_plot(df_alpha_out_sample[alpha], df_alpha_out_sample[columns_list])
            self.out_samples_plot(df_alpha_out_sample[alpha], df_alpha_out_sample[columns_list], mean_value)

    def in_samples_plot(self, alpha, returns):
        alphaname = alpha.name
        fig2 = plt.figure(constrained_layout=True, figsize=(22, 4))
        spec2 = gridspec.GridSpec(ncols=6, nrows=1, figure=fig2)
        f2_ax1 = fig2.add_subplot(spec2[0, 0])
        f2_ax2 = fig2.add_subplot(spec2[0, 1])
        f2_ax3 = fig2.add_subplot(spec2[0, 2])
        f2_ax4 = fig2.add_subplot(spec2[0, 3])
        f2_ax5 = fig2.add_subplot(spec2[0, 4])
        f2_ax6 = fig2.add_subplot(spec2[0, 5])
        f2_ax1.scatter(alpha, returns['t_1'], c='g')
        f2_ax2.scatter(alpha, returns['t_2'], c='g')
        f2_ax3.scatter(alpha, returns['t_3'], c='g')
        f2_ax4.scatter(alpha, returns['t_4'], c='g')
        f2_ax5.scatter(alpha, returns['t_5'], c='g')
        f2_ax6.scatter(alpha, returns['t_6'], c='g')
        f2_ax1.set_title(alphaname + ' t+1 in-sample')
        f2_ax2.set_title(alphaname + ' t+2 in-sample')
        f2_ax3.set_title(alphaname + ' t+3 in-sample')
        f2_ax4.set_title(alphaname + ' t+4 in-sample')
        f2_ax5.set_title(alphaname + ' t+5 in-sample')
        f2_ax6.set_title(alphaname + ' t+6 in-sample')
        f2_ax1.set_xlabel(alphaname)
        f2_ax2.set_xlabel(alphaname)
        f2_ax3.set_xlabel(alphaname)
        f2_ax4.set_xlabel(alphaname)
        f2_ax5.set_xlabel(alphaname)
        f2_ax6.set_xlabel(alphaname)
        f2_ax1.set_ylabel('t+1 return')
        f2_ax2.set_ylabel('t+2 return')
        f2_ax3.set_ylabel('t+3 return')
        f2_ax4.set_ylabel('t+4 return')
        f2_ax5.set_ylabel('t+5 return')
        f2_ax6.set_ylabel('t+6 return')
        plt.show()

    def out_samples_plot(self, alpha, returns, mean_value=0):
        alpha = alpha - mean_value
        ic_dict = self.ic_calculator(alpha, returns)
        alphaname = alpha.name
        fig2 = plt.figure(constrained_layout=True, figsize=(22, 4))
        spec2 = gridspec.GridSpec(ncols=6, nrows=1, figure=fig2)
        f2_ax1 = fig2.add_subplot(spec2[0, 0])
        f2_ax2 = fig2.add_subplot(spec2[0, 1])
        f2_ax3 = fig2.add_subplot(spec2[0, 2])
        f2_ax4 = fig2.add_subplot(spec2[0, 3])
        f2_ax5 = fig2.add_subplot(spec2[0, 4])
        f2_ax6 = fig2.add_subplot(spec2[0, 5])
        f2_ax1.scatter(alpha, returns['t_1'])
        f2_ax2.scatter(alpha, returns['t_2'])
        f2_ax3.scatter(alpha, returns['t_3'])
        f2_ax4.scatter(alpha, returns['t_4'])
        f2_ax5.scatter(alpha, returns['t_5'])
        f2_ax6.scatter(alpha, returns['t_6'])
        f2_ax1.set_title(alphaname + ' t+1 out-sample IC:%.3f' % ic_dict['t_1'])
        f2_ax2.set_title(alphaname + ' t+2 out-sample IC:%.3f' % ic_dict['t_2'])
        f2_ax3.set_title(alphaname + ' t+3 out-sample IC:%.3f' % ic_dict['t_3'])
        f2_ax4.set_title(alphaname + ' t+4 out-sample IC:%.3f' % ic_dict['t_4'])
        f2_ax5.set_title(alphaname + ' t+5 out-sample IC:%.3f' % ic_dict['t_5'])
        f2_ax5.set_title(alphaname + ' t+5 out-sample IC:%.3f' % ic_dict['t_6'])
        f2_ax1.set_xlabel(alphaname + ' %.3f' % (mean_value))
        f2_ax2.set_xlabel(alphaname + ' %.3f' % (mean_value))
        f2_ax3.set_xlabel(alphaname + ' %.3f' % (mean_value))
        f2_ax4.set_xlabel(alphaname + ' %.3f' % (mean_value))
        f2_ax5.set_xlabel(alphaname + ' %.3f' % (mean_value))
        f2_ax6.set_xlabel(alphaname + ' %.3f' % (mean_value))
        f2_ax1.set_ylabel('t+1 return')
        f2_ax2.set_ylabel('t+2 return')
        f2_ax3.set_ylabel('t+3 return')
        f2_ax4.set_ylabel('t+4 return')
        f2_ax5.set_ylabel('t+5 return')
        f2_ax6.set_ylabel('t+6 return')
        plt.show()

    def ic_calculator(self, alpha, returns):
        total_no_samples = len(alpha.index)
        positive_sginal_selected = alpha[alpha > 0]
        netgative_sginal_selected = alpha[alpha < 0]
        positive_sginal = returns.loc[positive_sginal_selected.index]
        netgative_sginal = returns.loc[netgative_sginal_selected.index]
        correct = (positive_sginal > 0).sum() + (netgative_sginal < 0).sum()
        ic_ratio = (correct / total_no_samples) * 2 - 1
        return dict(ic_ratio)

    def single_factor_optimisation(self,alhpa):
        pass
    def cases_to_try(self,params):
        pass

if __name__ == '__main__':
    # -- import data --

    data = pd.read_csv("binance_swap_kline.BTCUSD.csv", index_col=5)
    data.index = [datetime.datetime.fromtimestamp(int(ts) / 1000) for ts in data.index]
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data['Turnover'] = data['Volume'] * data['Close']

    # -- factors analysis --
    FA = FactorsAnalyzer(alpha101)
    FA.add_data(data, '15min')
    FA.initialization()
    FA.analysis()