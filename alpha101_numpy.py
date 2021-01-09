import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import time
np.seterr(divide='ignore', invalid='ignore')

class Alpha101_numpy(object):
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
        self.optimized_params = {}
        self.default_params['alpha_006'] = {'d_1':10}
        self.default_params['alpha_007'] = {'d_1':20,'d_2':7,'d_3':60,'d_4':7,'c_1':1}
        self.default_params['alpha_009'] = {'c_1' : 0,'d_1':1,'d_2':5,'d_3':1}
        self.default_params['alpha_012'] = {'d_1':1,'d_2':1}
        self.default_params['alpha_023'] = {'d_1':20,'c_1':20,'d_2':2,'c_2':0}
        self.default_params['alpha_024'] = {'d_1':100,'c_1':0.05,'d_2':3}
        self.default_params['alpha_026'] = {'d_1':5,'d_2':3}
        self.default_params['alpha_035'] = {'d_1':32,'d_2':16,'d_3':32}
        self.default_params['alpha_041'] = {'c_1':0.5}
        self.default_params['alpha_043'] = {'d_1':20,'d_2':7,'d_3':8}
        self.default_params['alpha_046'] = {'d_1':20,'d_2':10,'d_3':10,'c_1':0.25,'c_2':1,'c_3':0,'c_4':1,'c_5':1}
        self.default_params['alpha_049'] = {'d_1':20,'d_2':10,'d_3':10,'c_1':0.1,'c_2':1,'d_4':1}
        self.default_params['alpha_051'] = {'d_1':20,'d_2':10,'d_3':10,'c_1':0.05,'c_2':1,'d_4':1}
        self.default_params['alpha_053'] = {'d_1':9}
        self.default_params['alpha_054'] = {'c_1':5}
        self.default_params['alpha_084'] = {'d_1':15,'d_2':21,'d_3':5}
        self.default_params['alpha_101'] = {'c_1':0.001}

        self.optimized_params['alpha_006'] = {'d_1':120} # 0.009
        # self.optimized_params['alpha_007'] = {'d_1':20,'d_2':7,'d_3':60,'d_4':7,'c_1':1}
        # self.optimized_params['alpha_009'] = {'c_1' : 0,'d_1':1,'d_2':5,'d_3':1}
        self.optimized_params['alpha_012'] = {'d_1':30,'d_2':50} # 0.0209
        # self.optimized_params['alpha_023'] = {'d_1':20,'c_1':20,'d_2':2,'c_2':0}
        self.optimized_params['alpha_024'] = {'d_1':10,'c_1':1,'d_2':120} # -0.0343
        self.optimized_params['alpha_026'] = {'d_1':5,'d_2':50} # 0.116
        # self.optimized_params['alpha_035'] = {'d_1':32,'d_2':16,'d_3':32}
        self.optimized_params['alpha_041'] = {'c_1':0.5} # -0.0196
        # self.optimized_params['alpha_043'] = {'d_1':20,'d_2':7,'d_3':8}
        # self.optimized_params['alpha_046'] = {'d_1':20,'d_2':10,'d_3':10,'c_1':0.25,'c_2':1,'c_3':0,'c_4':1,'c_5':1}
        # self.optimized_params['alpha_049'] = {'d_1':20,'d_2':10,'d_3':10,'c_1':0.1,'c_2':1,'d_4':1}
        # self.optimized_params['alpha_051'] = {'d_1':20,'d_2':10,'d_3':10,'c_1':0.05,'c_2':1,'d_4':1}
        self.optimized_params['alpha_053'] = {'d_1':9} # -0.0097
        self.optimized_params['alpha_054'] = {'c_1':1} # -0.0196
        # self.optimized_params['alpha_084'] = {'d_1':15,'d_2':21,'d_3':5}
        self.optimized_params['alpha_101'] = {'c_1':0.1} #0.0254
        self.default_params = self.optimized_params

    def input_data(self,data,tf=5):
        """
        Data is dataframe object with Date, Open, High, Low, Close, Volume, Turnover, VWAP

        """
        self.data = data
        self.open_price = self.first_value(data.Open.values,tf)
        self.high_price = self.high_value(data.High.values,tf)
        self.low_price = self.low_value(data.Low.values,tf)
        self.close_price = self.last_value(self.data.Close.values,tf)
        self.volume = self.sum_value(data.Volume.values,tf)
        self.turnover = self.sum_value(data.Turnover.values,tf)
        self.delayed_close = self.delay(self.close_price,tf)
        self.returns = self.close_price/self.delayed_close-1
        self.vwap = self.turnover/self.volume


    def first_value(self,a,window):
        def rolling_window(a, window):
            # 1d array rolling
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        x = rolling_window(a,window)
        na = np.empty((window-1))
        na[:] = np.nan
        return np.concatenate((na,x[:,0]))

    def last_value(self,a,window):
        def rolling_window(a, window):
            # 1d array rolling
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        x = rolling_window(a,window)
        na = np.empty((window-1))
        na[:] = np.nan
        return np.concatenate((na,x[:,-1]))

    def high_value(self,a,window):
        def rolling_window(a, window):
            # 1d array rolling
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        x = rolling_window(a,window)
        na = np.empty((window-1))
        na[:] = np.nan
        return np.concatenate((na,np.max(x,axis=1)))

    def low_value(self,a,window):
        def rolling_window(a, window):
            # 1d array rolling
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        x = rolling_window(a,window)
        na = np.empty((window-1))
        na[:] = np.nan
        return np.concatenate((na,np.min(x,axis=1)))
    def sum_value(self,a,window):
        def rolling_window(a, window):
            # 1d array rolling
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        x = rolling_window(a,window)
        na = np.empty((window-1))
        na[:] = np.nan
        return np.concatenate((na,np.sum(x,axis=1)))

    def correlation(self,x,y,d):
        return pd.Series(x).rolling(d).corr(pd.Series(y)).values

    def covariance(self,x,y,d):
        return pd.Series(x).rolling(d).cov(pd.Series(y)).values

    def scale(self,x,a=1):
        # waiting for series change
        x_abs = np.abs(x)
        x_sum = np.sum(x_abs)
        return x/x_sum

    def delta(self,x,d):
        x1 = np.roll(x,d)
        x1[:d]=np.nan
        return x-x1

    def signedpower(self,x,a):
        return np.power(x,a)

    def decay_linear(self,x, d):
        # wait for series change
        weight = np.arange(d)+1
        weight = self.scale(weight)
        return np.average(x[-d:],weights=weight)

    def delay(self,x,d):
        x = np.roll(x,d)
        x[:d]=np.nan
        return x

    def to_rank_ss(self,x):
        # result[i] is the rank of x[i] in x
        return np.sum(np.less(x, x.iat[-1]))

    def ts_rank(self,x,d):
        #def to_rank(x):
        #    return np.sum(np.less(x.values,x.values[-1]))
        #return pd.Series(x).rolling(d).apply(to_rank).values/d
        def rolling_window(a, d):
            # 1d array rolling
            shape = a.shape[:-1] + (a.shape[-1] - d + 1, d)
            strides = a.strides + (a.strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        x = rolling_window(x,d)
        na = np.empty((d-1))
        na[:] = np.nan
        return np.concatenate((na,np.sum(np.less(x,x[:,-1].reshape(-1,1)),axis=1) / d))

    def ts_max(self,x,d):
        return pd.Series(x).rolling(d).max().values
    def ts_min(self,x,d):
        return pd.Series(x).rolling(d).min().values

    def ts_argmin(self,x,d):
        return d - pd.Series(x).rolling(d).apply(np.argmin).values

    def ts_argmax(self,x,d):
        return d - pd.Series(x).rolling(d).apply(np.argmax).values

    def sum_x(self,x,d):
        return pd.Series(x).rolling(d).sum().values

    def adv(self,d):
        return pd.Series(self.turnover).rolling(d).mean().values
# ----------------------------alpha -------------------------------------

    def alpha_006(self,params=None):
        # Alpha#6: (-1 * correlation(open, volume, 10))
        if params==None:
            params=self.default_params['alpha_006']
        d_1 = params['d_1']
        return -1*self.correlation(self.open_price,self.volume,d_1)

    def alpha_007(self,params=None):
        # Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
        if params==None:
            params=self.default_params['alpha_007']
        d_1=params['d_1']
        d_2=params['d_2']
        d_3=params['d_3']
        d_4=params['d_4']
        c_1=params['c_1']
        condition_A = (self.adv(d_1) < self.volume)*1
        result_A = -1*self.ts_rank(np.abs(self.delta(self.close_price,d_2)),d_3)*np.sign(self.delta(self.close_price,d_4))
        return result_A*condition_A + (1-condition_A)*(-1*c_1)


    def alpha_009(self,params=None):
        # Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?
        # delta(close, 1) : (-1 * delta(close, 1))))
        if params==None:
            params=self.default_params['alpha_009']
        c_1 = params['c_1']
        d_1=params['d_1']
        d_2=params['d_2']
        d_3=params['d_3']

        condition_A = (c_1 < self.ts_min(self.delta(self.close_price,d_1),d_2))*1
        result_A = self.delta(self.close_price,d_1)

        condition_B = (1-condition_A)*(self.ts_min(self.delta(self.close_price,d_1),d_2) < c_1)*1
        result_B = self.delta(self.close_price,d_1)

        condition_C = (1-condition_A)*(1-condition_B)
        result_C = -1*self.delta(self.close_price,d_3)

        return condition_A*result_A+condition_B*result_B+condition_C*result_C
    def alpha_012(self,params=None):
        # Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        if params==None:
            params=self.default_params['alpha_012']
        d_1=params['d_1']
        d_2=params['d_2']
        return np.sign(self.delta(self.volume,d_1))*(-1*self.delta(self.close_price,d_2))

#     def alpha_021(self):
#         # Alpha#21: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,
#         # 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /
#         # adv20) == 1)) ? 1 : (-1 * 1))))
#         pass

    def alpha_023(self,params=None):
        # Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
        if params==None:
            params=self.default_params['alpha_023']
        d_1=params['d_1']
        d_2=params['d_2']
        c_1=params['c_1']
        c_2=params['c_2']
        condition_A = (self.sum_x(self.high_price,d_1)/c_1 < self.high_price)*1
        result_A = -1*self.delta(self.high_price,d_2)

        conditon_B = (1-condition_A)
        result_B = c_2
        return condition_A*result_A + conditon_B*result_B

    def alpha_024(self,params=None):
        # Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||
        # ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,
        # 100))) : (-1 * delta(close, 3)))
        if params==None:
            params=self.default_params['alpha_024']
        d_1=params['d_1']
        d_2=params['d_2']
        c_1=params['c_1']
        condition_A = ((self.delta(self.sum_x(self.close_price,d_1)/d_1,d_1)/self.delay(self.close_price,d_1))<=c_1)*1
        result_A = -1*(self.close_price-self.ts_min(self.close_price,d_1))

        condition_B = (1-condition_A)
        result_B = -1*self.delta(self.close_price,d_2)

        return condition_A*result_A + condition_B*result_B

    def alpha_026(self,params=None):
        # Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        # print(self.ts_rank_series(self.volume,5),self.ts_rank_series(self.high_price,5))
        if params==None:
            params=self.default_params['alpha_026']
        d_1=params['d_1']
        d_2=params['d_2']
        a = self.correlation(self.ts_rank(self.volume,d_1),self.ts_rank(self.high_price,d_1),d_1)
        b = self.ts_max(a,d_2)
        return -1*b

#     def alpha_028(self):
#         # Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
#         pass

#     def alpha_032(self):
#         # Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
#         pass

    def alpha_035(self,params=None):
        # Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
        if params==None:
            params=self.default_params['alpha_035']
        d_1=params['d_1']
        d_2=params['d_2']
        d_3=params['d_3']
        a = self.ts_rank(self.volume,d_1)
        b = 1 - self.ts_rank(((self.close_price+self.high_price)-self.low_price),d_2)
        c = 1 - self.ts_rank(self.returns,d_3)
        return a*b*c

    def alpha_041(self,params=None):
        # Alpha#41: (((high * low)^0.5) - vwap)
        if params==None:
            params=self.default_params['alpha_041']
        c_1=params['c_1']
        return np.power(self.high_price*self.low_price,c_1)-self.vwap

    def alpha_043(self,params=None):
        # Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
        if params==None:
            params=self.default_params['alpha_043']
        d_1=params['d_1']
        d_2=params['d_2']
        d_3=params['d_3']
        return self.ts_rank(self.volume/self.adv(d_1),d_1)*self.ts_rank(-1*self.delta(self.close_price,d_2),d_3)

    def alpha_046(self,params=None):
        # Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?
        # (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :
        # ((-1 * 1) * (close - delay(close, 1)))))
        if params==None:
            params=self.default_params['alpha_046']
        d_1=params['d_1']
        d_2=params['d_2']
        d_3=params['d_3']
        c_1=params['c_1']
        c_2=params['c_2']
        c_3=params['c_3']
        c_4=params['c_4']
        c_5=params['c_5']
        a = (self.delay(self.close_price,d_1)-self.delay(self.close_price,d_2))/d_2
        b = (self.delay(self.close_price,d_3)-self.close_price)/d_3

        condition_A = (c_1 < (a-b))*1
        result_A = -1*c_2

        condition_B = (1-condition_A)*((a-b)<c_3)
        result_B = c_4

        condition_C = (1-condition_A)*(1-condition_B)
        result_C = -1*c_5*(self.close_price-self.delay(self.close_price,1))

        return condition_A*result_A+condition_B*result_B+condition_C*result_C

    def alpha_049(self,params=None):
        # Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *
        # 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        if params==None:
            params=self.default_params['alpha_049']
        d_1=params['d_1']
        d_2=params['d_2']
        d_3=params['d_3']
        d_4=params['d_4']
        c_1=params['c_1']
        c_2=params['c_2']
        a = (self.delay(self.close_price,d_1)-self.delay(self.close_price,d_2))/d_2
        b = (self.delay(self.close_price,d_3)-self.close_price)/d_3

        condition_A = ((a-b) < -1*c_1)*1
        result_A = c_2

        condition_B = (1-condition_A)
        result_B = -1*(self.close_price-self.delay(self.close_price,d_4))

        return condition_A*result_A+condition_B*result_B

    def alpha_051(self,params=None):
        # Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *
        # 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        if params==None:
            params=self.default_params['alpha_051']
        d_1=params['d_1']
        d_2=params['d_2']
        d_3=params['d_3']
        d_4=params['d_4']
        c_1=params['c_1']
        c_2=params['c_2']
        a = (self.delay(self.close_price,d_1)-self.delay(self.close_price,d_2))/d_2
        b = (self.delay(self.close_price,d_3)-self.close_price[-1])/d_3

        condition_A = ((a-b) < -1*c_1)*1
        result_A = c_2
        condition_B = (1-condition_A)
        result_B = -1*(self.close_price-self.delay(self.close_price,d_4))
        return condition_A*result_A + condition_B*result_B

    def alpha_053(self,params=None):
        # Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        if params==None:
            params=self.default_params['alpha_053']
        d_1=params['d_1']
        a = self.close_price-self.low_price
        b = self.high_price - self.close_price
        x = (a-b)/(a)
        return -1*self.delta(x,d_1)

    def alpha_054(self,params=None):
        # Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        if params==None:
            params=self.default_params['alpha_054']
        c_1=params['c_1']
        return (-1*(self.low_price-self.close_price)*(np.power(self.open_price,c_1)))/((self.low_price - self.high_price)*(np.power(self.close_price,c_1)))

    def alpha_084(self,params=None):
        # Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
        if params==None:
            params=self.default_params['alpha_084']
        d_1=params['d_1']
        d_2=params['d_2']
        d_3=params['d_3']
        x = self.ts_rank((self.vwap-self.ts_max(self.vwap,d_1)),d_2)
        a = self.delta(self.close_price,d_3)
        return self.signedpower(x,a)

    def alpha_101(self,params=None):
        # Alpha#101: ((close - open) / ((high - low) + .001))
        if params==None:
            params=self.default_params['alpha_101']
        c_1=params['c_1']
        return ((self.close_price-self.open_price)/(self.high_price-self.low_price+c_1))

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
        for alpha in self.default_params.keys():
            func = getattr(self, alpha)
            val = func()
            alpha_values.append(val)
        return alpha_values

    def compute_single_alpha(self,alpha,params = None):
        func = getattr(self, alpha)
        if params is None:
            val = func()
        else:
            val = func(params)
        return val



class FactorsAnalyzer_numpy(object):
    def __init__(self,alpha101):
        self.alphaModel = alpha101()
        self.default_params = self.alphaModel.default_params
        self.data = None
        self.resampled_data = None
        self.alpha_list = None
        self.pre_data_analysis = None
        self.resample_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Turnover':'sum'
        }
        self.alpha_data = None

    def add_data(self,data,freq= '60min'):
        self.data = data
        self.resampling(freq)

    def resampling(self,freq = '60min'):
        data_hour = self.data# .resample(freq).agg(self.resample_dict)
        # data_hour['VWAP'] = data_hour['Turnover']/data_hour['Volume']
        self.resampled_data = data_hour
        self.pre_optimization_initialization()
    def pre_optimization_initialization(self):
        data_analysis = self.resampled_data
        data_analysis['t_1'] =data_analysis['Close'].shift(-1)/data_analysis['Close'].shift(-6)-1
        self.pre_data_analysis = data_analysis

    def initialization(self):
        self.alphaModel.input_data(self.resampled_data)

        alpha_data = {}
        data_length = len(self.resampled_data.index)
        for i in range(data_length-120):
            slice_data = self.resampled_data.iloc[i:i+121]
            date = slice_data.index[-1]
            self.alphaModel.input_data(slice_data)
            alpha_val = self.alphaModel.compute_all()
            alpha_data[date] = alpha_val
        alpha_df = pd.DataFrame.from_dict(alpha_data,orient='index')
        alpha_df.columns = self.alphaModel.alpha_list
        self.alpha_list = self.alphaModel.alpha_list
        data_analysis = self.resampled_data
        data_analysis['t_1'] =data_analysis['Close'].shift(-1)/data_analysis['Close']-1
        data_analysis['t_2'] =data_analysis['Close'].shift(-2)/data_analysis['Close']-1
        data_analysis['t_3'] =data_analysis['Close'].shift(-3)/data_analysis['Close']-1
        data_analysis['t_4'] =data_analysis['Close'].shift(-4)/data_analysis['Close']-1
        data_analysis['t_5'] =data_analysis['Close'].shift(-5)/data_analysis['Close']-1
        data_analysis['t_6'] =data_analysis['Close'].shift(-6)/data_analysis['Close']-1
        df = data_analysis.join(alpha_df)
        self.alpha_data = df

    def analysis(self):
        # columns_list = ['t_1','t_2','t_3','t_4','t_5','t_6']
        columns_list = ['t_1']
        for alpha in self.alpha_list:
            alpha_name = [alpha]
            columns_selected = alpha_name + columns_list
            df_alpha = self.alpha_data[columns_selected]
            df_alpha.dropna(inplace=True)

            total_sample = len(df_alpha.index)
            in_sample_len = round(total_sample*0.6)
            df_alpha_in_sample = df_alpha.iloc[:in_sample_len]
            df_alpha_out_sample = df_alpha.iloc[in_sample_len:]


            self.in_samples_plot(df_alpha_in_sample[alpha],df_alpha_in_sample[columns_list])
            mean_value = np.nanmean(df_alpha_in_sample[alpha])

            self.out_samples_plot(df_alpha_out_sample[alpha],df_alpha_out_sample[columns_list])
            self.out_samples_plot(df_alpha_out_sample[alpha],df_alpha_out_sample[columns_list],mean_value)
            plt.show()

    def in_samples_plot(self,alpha,returns):
        alphaname = alpha.name
        information_correlation_dict = self.information_correlation(alpha,returns)
        information_coefficient_dict = self.information_coefficient(alpha,returns)
        fig2 = plt.figure(constrained_layout=True,figsize=(22,4))
        spec2 = gridspec.GridSpec(ncols=6, nrows=1, figure=fig2)
        f2_ax1 = fig2.add_subplot(spec2[0, 0])
        f2_ax2 = fig2.add_subplot(spec2[0, 1])
        f2_ax3 = fig2.add_subplot(spec2[0, 2])
        f2_ax4 = fig2.add_subplot(spec2[0, 3])
        f2_ax5 = fig2.add_subplot(spec2[0, 4])
        f2_ax6 = fig2.add_subplot(spec2[0, 5])
        f2_ax1.scatter(alpha,returns['t_1'],c='g')
        f2_ax1.set_title(alphaname+' t+1 in-sample IC:%.3f'%information_coefficient_dict['t_1'])
        f2_ax1.set_xlabel(alphaname+' corr:%.3f' %(information_correlation_dict['t_1']))
        f2_ax1.set_ylabel('t+1 return')



    def out_samples_plot(self,alpha,returns,mean_value=0):
        alpha = alpha - mean_value
        information_coefficient_dict = self.information_coefficient(alpha,returns)
        information_correlation_dict = self.information_correlation(alpha,returns)
        alphaname = alpha.name
        fig2 = plt.figure(constrained_layout=True,figsize=(22,4))
        spec2 = gridspec.GridSpec(ncols=6, nrows=1, figure=fig2)
        f2_ax1 = fig2.add_subplot(spec2[0, 0])
        f2_ax2 = fig2.add_subplot(spec2[0, 1])
        f2_ax3 = fig2.add_subplot(spec2[0, 2])
        f2_ax4 = fig2.add_subplot(spec2[0, 3])
        f2_ax5 = fig2.add_subplot(spec2[0, 4])
        f2_ax6 = fig2.add_subplot(spec2[0, 5])
        f2_ax1.scatter(alpha,returns['t_1'])
        f2_ax1.set_title(alphaname+' t+1 out-sample IC:%.3f'%information_coefficient_dict['t_1'])
        f2_ax1.set_xlabel(alphaname+' %.3f corr:%.3f' %(mean_value,information_correlation_dict['t_1']))
        f2_ax1.set_ylabel('t+1 return')


    def information_coefficient(self,alpha,returns):
        total_no_samples = len(alpha.index)
        positive_sginal_selected = alpha[alpha>0]
        netgative_sginal_selected = alpha[alpha<0]
        positive_sginal = returns.loc[positive_sginal_selected.index]
        netgative_sginal = returns.loc[netgative_sginal_selected.index]
        correct = (positive_sginal> 0).sum() + (netgative_sginal < 0).sum()
        ic_ratio = (correct/total_no_samples)*2-1
        return dict(ic_ratio)

    def information_correlation(self,alpha,returns):
        df = returns.join(alpha)
        return dict(df.corr()[alpha.name])

    def information_correlation_df(self,alpha,returns):
        df = returns.join(alpha)
        return df.corr()[alpha.columns].loc[returns.columns].transpose()

    def cases_to_be_tried(self,params):
        d = list(np.array([1,3,5,10,20,30,50,70,90,120]))
        c_0 = [0.001,0.005,0.01,0.025,0.05,0.075,0.1]
        c_1 = [0.075,0.1,0.2,0.5,0.7,0.9,1]
        c_2 = [1,3,5,10,20,30,50,70,90,120]
        cases_dict = {}
        param_list = []
        for param in params.keys():
            if param[0] == 'd':
                param_list.append(d)
            elif param[0] == 'c' and params[param] < 0.05:
                param_list.append(c_0)
            elif param[0] == 'c' and params[param] >= 0.05 and params[param] < 1:
                param_list.append(c_1)
            elif param[0] == 'c' and params[param] >= 1:
                param_list.append(c_2)
        combination = [p for p in itertools.product(*param_list)]
        return combination

    def signle_factor_optimizer_testing(self,alpha_name):
        params = self.default_params[alpha_name]
        combinations = self.cases_to_be_tried(params)
        keys = list(params.keys())
        optimized_data = []
        # done_case = 0
        cases_number = len(combinations)
        for comb in combinations[:1]:
            # print(done_case/cases_number)
            new_params = dict(zip(keys, comb))
            information_coefficient_dict, information_correlation_dict = self._optimizer(alpha_name,new_params)
            optimized_data.append([new_params,information_coefficient_dict, information_correlation_dict])

            # done_case += 1

        idx = 0
        final_df = pd.DataFrame()
        for data in optimized_data:
            params = data[0]
            coeff = data[1]
            correl = data[2]
            coeff_df = pd.DataFrame(coeff,index=[idx])[['t_1','t_2','t_3','t_4','t_5','t_6']]
            coeff_df.columns = ['coeff t_1','coeff t_2','coeff t_3','coeff t_4','coeff t_5','coeff t_6']
            coeff_df['coeff max'] = np.nanmax(np.abs(coeff_df))
            correl_df = pd.DataFrame(correl,index=[idx])[['t_1','t_2','t_3','t_4','t_5','t_6']]
            correl_df.columns = ['correl t_1','correl t_2','correl t_3','correl t_4','correl t_5','correl t_6']
            correl_df['correl max'] = np.nanmax(np.abs(correl_df))
            params_df = pd.DataFrame([[params]],index=[idx],columns=['params'])
            temp_df = pd.concat([params_df,coeff_df,correl_df],axis=1)
            final_df = pd.concat([final_df,temp_df],axis=0)
            idx +=1
        return final_df

    def signle_factor_optimizer(self,alpha_name):
        params = self.default_params[alpha_name]
        combinations = self.cases_to_be_tried(params)
        keys = list(params.keys())
        optimized_data = []
        # done_case = 0
        cases_number = len(combinations)
        for comb in combinations[:200]:
            # print(done_case/cases_number)
            new_params = dict(zip(keys, comb))
            information_coefficient_dict, information_correlation_dict = self._optimizer(alpha_name,new_params)
            optimized_data.append([new_params,information_coefficient_dict, information_correlation_dict])

            # done_case += 1

        idx = 0
        final_df = pd.DataFrame()
        for data in optimized_data:
            params = data[0]
            coeff = data[1]
            correl = data[2]
            coeff_df = pd.DataFrame(coeff,index=[idx])[['t_1','t_2','t_3','t_4','t_5','t_6']]
            coeff_df.columns = ['coeff t_1','coeff t_2','coeff t_3','coeff t_4','coeff t_5','coeff t_6']
            coeff_df['coeff max'] = np.nanmax(np.abs(coeff_df))
            correl_df = pd.DataFrame(correl,index=[idx])[['t_1','t_2','t_3','t_4','t_5','t_6']]
            correl_df.columns = ['correl t_1','correl t_2','correl t_3','correl t_4','correl t_5','correl t_6']
            correl_df['correl max'] = np.nanmax(np.abs(correl_df))
            params_df = pd.DataFrame([[params]],index=[idx],columns=['params'])
            temp_df = pd.concat([params_df,coeff_df,correl_df],axis=1)
            final_df = pd.concat([final_df,temp_df],axis=0)
            idx +=1
        return final_df

    def signle_factor_optimizer_v2(self,alpha_name):
        params = self.default_params[alpha_name]
        combinations = self.cases_to_be_tried(params)
        keys = list(params.keys())
        optimized_data = []
        cases_number = len(combinations)
        combinations_list = []
        for comb in combinations:
            new_params = dict(zip(keys, comb))
            combinations_list.append(new_params)
        information_correlation_df= self._optimizer_v2(alpha_name,combinations_list)
        return information_correlation_df

    def _optimizer_v2(self,alpha_name,params):
        data_length = len(self.resampled_data.index)

        slice_data = self.resampled_data
        date = slice_data.index[-1]
        self.alphaModel.input_data(slice_data)
        temp_alpha_data = {}
        temp_param_data = {}
        idx = 0
        for param in params:
            alpha_val = self.alphaModel.compute_single_alpha(alpha_name,param)
            temp_alpha_data[idx] = alpha_val
            temp_param_data[idx] = param
            idx +=1
        #print(temp_alpha_data)
        alpha_df = pd.DataFrame.from_dict(temp_alpha_data)
        alpha_df.index = slice_data.index
        # print(alpha_df)
        # alpha_df.to_csv('alpha54data.csv')
        self.alpha_list = self.alphaModel.alpha_list
        df = self.pre_data_analysis.copy().join(alpha_df)
        columns_list = ['t_1']#,'t_2','t_3','t_4','t_5','t_6']
        columns_selected = list(alpha_df.columns)+ columns_list
        df_alpha = df[columns_selected].copy()

        # df_alpha.dropna(inplace=True)
        # df_alpha.to_csv('alpha26data2.csv')
        # alpha = df_alpha[alpha_df.columns]
#         returns = df_alpha[columns_list]
#         ic_dict = {}
#         for alpha_case in alpha_df.columns:
#             alpha = df_alpha[alpha_case]
#             temp_dict = self.information_correlation_df(alpha,returns)
#             ic_dict = {**ic_dict,**temp_dict}
#         information_correlation_df = ic_dict
        alpha = df_alpha[alpha_df.columns]
        returns = df_alpha[columns_list]
        information_correlation_df = self.information_correlation_df(alpha,returns)
        param_df  = pd.DataFrame.from_dict(temp_param_data,orient='index')
        return information_correlation_df.join(param_df)

    def _optimizer(self,alpha_name,params):
        data_length = len(self.resampled_data.index)
        alpha_data = {}

        slice_data = self.resampled_data
        date = slice_data.index[-1]
        self.alphaModel.input_data(slice_data)
        alpha_val = self.alphaModel.compute_single_alpha(alpha_name,params)
        alpha_data[date] = alpha_val

        alpha_df = pd.DataFrame(alpha_val)
        alpha_df.index = slice_data.index
        alpha_df.columns = [alpha_name]
        self.alpha_list = self.alphaModel.alpha_list
        data_analysis = self.resampled_data
        data_analysis['t_1'] =data_analysis['Close'].shift(-1)/data_analysis['Close']-1
        data_analysis['t_2'] =data_analysis['Close'].shift(-2)/data_analysis['Close']-1
        data_analysis['t_3'] =data_analysis['Close'].shift(-3)/data_analysis['Close']-1
        data_analysis['t_4'] =data_analysis['Close'].shift(-4)/data_analysis['Close']-1
        data_analysis['t_5'] =data_analysis['Close'].shift(-5)/data_analysis['Close']-1
        data_analysis['t_6'] =data_analysis['Close'].shift(-6)/data_analysis['Close']-1
        df = data_analysis.join(alpha_df)

        columns_list = ['t_1','t_2','t_3','t_4','t_5','t_6']
        columns_selected = [alpha_name]+ columns_list

        df_alpha = df[columns_selected].copy()
        df_alpha.dropna(inplace=True)

        alpha = df_alpha[alpha_name]
        returns = df_alpha[columns_list]
        information_coefficient_dict = self.information_coefficient(alpha,returns)
        information_correlation_dict = self.information_correlation(alpha,returns)

        return information_coefficient_dict, information_correlation_dict

if __name__ == '__main__':
    # -- import data --

    data = pd.read_csv("binance_swap_kline.BTCUSD.csv",index_col=5)
    data.index = [datetime.datetime.fromtimestamp(int(ts)/1000) for ts in data.index]
    data.columns = ['Open','High','Low','Close','Volume']
    data['Turnover'] = data['Volume'] *data['Close']

    # setup model
    FA_numpy = FactorsAnalyzer_numpy(Alpha101_numpy)
    FA_numpy.add_data(data,'15min')

    # optimisation for params < 5
    alpha_list_opt = []
    for alpha in params.keys():
        if len(params[alpha])<5:
            alpha_list_opt.append(alpha)

    optimized_data_dict = {}
    for alpha in alpha_list_opt:
        optimized_data_dict[alpha] = FA_numpy.signle_factor_optimizer(alpha)
        print(alpha + ' Done~!!!')
        optimized_data_dict[alpha].to_excel(alpha+'.xlsx')