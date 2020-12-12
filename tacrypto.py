# -*- coding: utf-8 -*-
'''
File: talibcrypto.py
File Created: Wednesday, 9th December 2020 4:54:37 pm
Author: Hongzoeng Ng (hongzoengng@jiupaicapitalibl.net)
-----
Last Modified: Wednesday, 9th December 2020 5:00:52 pm
CopyRigth @ Hongzoeng Ng
'''
import talib
import numpy as np
import pandas as pd


class talibCrypto(object):
    def __init__(self, minute_datalibframe):
        self.df_cols = ['open', 'high', 'low', 'close', 'volume', 'time', 'avg_price']
        self.df = minute_datalibframe
        self.talib_function = [
            self.bband, self.dema, self.ema, self.kama, self.ma, self.mama, self.midpoint,
            self.midprice, self.sar, self.sma, self.t3, self.tema, self.trima, self.wma,
            self.plusdi, self.minusdi, self.pmdi, self.adx, self.adx_pct, self.pmdi_aadx
        ]

    
    """
    Overlap Study
    """
    # 1. Bollinger Bands
    def bband(self):
        upperband, middleband, lowerband = talib.BBANDS(self.df['close'], timeperiod=50, nbdevup=10, nbdevdn=10, matype=7)
        real = (upperband - self.df['avg_price']) / (self.df['avg_price'] - lowerband)
        real.name = 'BBAND'
        return real
    
    # 2. Double Exponential Moving Average
    def dema(self):
        dema1 = talib.DEMA(self.df['close'], timeperiod=3)
        dema2 = talib.DEMA(self.df['close'], timeperiod=2)
        real = dema2 - dema1
        real.name = 'DEMA'
        return real
    
    # 3. Exponential Moving Average
    def ema(self):
        ema1 = talib.EMA(self.df['close'], timeperiod=3)
        ema2 = talib.EMA(self.df['close'], timeperiod=2)
        real = ema2 - ema1
        real.name = 'EMA'
        return real
    
    # 4. Kaufman Adaptive Moving Average
    def kama(self):
        kama1 = talib.KAMA(self.df['close'], timeperiod=3)
        kama2 = talib.KAMA(self.df['close'], timeperiod=2)
        real = kama2 - kama1
        real.name = 'KAMA'
        return real
    
    # 5. Moving average
    def ma(self):
        ma1 = talib.MA(self.df['close'], timeperiod=4, matype=4)
        ma2 = talib.MA(self.df['close'], timeperiod=2, matype=4)
        real = ma2 - ma1
        real.name = 'MA'
        return real
    
    # 6. MESA Adaptive Moving Average
    def mama(self):
        mama, fama = talib.MAMA(self.df['close'], fastlimit=0.9, slowlimit=0.01)
        real = fama - mama
        real.name = 'MAMA'
        return real
    
    # 7. MidPoint over period
    def midpoint(self):
        midp1 = talib.MIDPOINT(self.df['close'], timeperiod=10)
        midp2 = talib.MIDPOINT(self.df['close'], timeperiod=3)
        real = midp1 - midp2
        real.name = 'MIDPOINT'
        return real
    
    # 8. Midpoint Price over period
    def midprice(self):
        midp1 = talib.MIDPRICE(self.df['high'], self.df['low'], timeperiod=9)
        midp2 = talib.MIDPRICE(self.df['high'], self.df['low'], timeperiod=3)
        real = midp1 - midp2
        real.name = 'MIDPRICE'
        return real
    
    # 9. Parabolic SAR
    def sar(self):
        sar = talib.SAR(self.df['high'], self.df['low'], acceleration=0.4, maximum=1)
        real = self.df['close'] / sar - 1
        real.name = 'SAR'
        return real
    
    # 10. Simple Moving Average
    def sma(self):
        sma1 = talib.SMA(df['close'], timeperiod=5)
        sma2 = talib.SMA(df['close'], timeperiod=10)
        real = sma2 - sma1
        real.name = 'SMA'
        return real
    
    # 11. Triple Exponential Moving Average (T3)
    def t3(self):
        t3_1 = talib.T3(self.df['close'], timeperiod=4, vfactor=0)
        t3_2 = talib.T3(self.df['close'], timeperiod=3, vfactor=0)
        real = t3_1 - t3_2
        real.name = 'T3'
        return real
    
    # 12. TEMA - Triple Exponential Moving Average
    def tema(self):
        tema1 = talib.TEMA(self.df['close'], timeperiod=4)
        tema2 = talib.TEMA(self.df['close'], timeperiod=3)
        real = tema2 - tema1
        real.name = 'TEMA'
        return real
    
    # 13. Triangular Moving Average
    def trima(self):
        trima1 = talib.TRIMA(self.df['close'], timeperiod=3)
        trima2 = talib.TRIMA(self.df['close'], timeperiod=2)
        real = trima2 - trima1
        real.name = 'TRIMA'
        return real
    
    # 14. Weighted Moving Average
    def wma(self):
        wma1 = talib.WMA(self.df['close'], timeperiod=3)
        wma2 = talib.WMA(self.df['close'], timeperiod=2)
        real = wma2 - wma1
        real.name = 'WMA'
        return real
    

    """
    Momentum Indicator
    """
    # 15. Plus Directional Indicator (+DI)
    def plusdi(self):
        pdi1 = talib.PLUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=10)
        pdi2 = talib.PLUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=5)
        real = pdi1 / pdi2
        real.name = 'PLUS_DI'
        return real

    # 16. Minus Directional Indicator (-DI)
    def minusdi(self):
        mdi1 = talib.MINUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=3)
        mdi2 = talib.MINUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=2)
        real = mdi1 - mdi2
        real.name = 'MINUS_DI'
        return real
    
    # 17. Plus-Minus Directional Indicator [self-developed]
    def pmdi(self):
        pdi1 = talib.PLUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=15)
        pdi2 = talib.PLUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=7)
        deltalib_pdi = pdi1 / pdi2

        mdi1 = talib.MINUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=15)
        mdi2 = talib.MINUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=7)
        deltalib_mdi = mdi1 - mdi2

        real = deltalib_pdi.fillna(0).values / np.linalg.norm(deltalib_pdi.fillna(0).values) - \
            deltalib_mdi.fillna(0).values / np.linalg.norm(deltalib_mdi.fillna(0).values)
        real = pd.Series(real, index=deltalib_pdi.index)
        real.name = 'PM_DI'
        return real
    
    # 18. Average Directional Movement Index
    def adx(self):
        real = talib.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=5)
        real.name = 'ADX'
        return real

    # 19. Average Directional Movement Index Percentalibge Change  [self-developed]
    def adx_pct(self):
        adx = talib.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=5)
        real = adx.pct_change(5)
        real.name = 'ADX_PCTCHNG'
        return real
    
    # 20. Plus-Minus DI Adjusted ADX [self-developed]
    def pmdi_aadx(self): 
        pdi = talib.PLUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=6)
        mdi = talib.MINUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=6)
        adx = talib.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=3)
        real = adx.pct_change(5) * (mdi - pdi)
        real.name = 'PMDI_AADX'
        return real
    
    # 21. Average Directional Movement Index Rating
    def axdr(self):
        adxr = talib.ADXR(self.df['high'], self.df['low'], self.df['close'], timeperiod=5)
        real = adxr - self.df['low']
        real.name = 'AXDR'
        return real
    
    # 22. Absolute Price Oscillator
    def apo(self):
        real = talib.APO(self.df['close'], fastperiod=5, slowperiod=10, matype=5)
        real.name = 'APO'
        return real
    
    # 23. Aroon Oscillator
    def aroonosc(self):
        real = talib.AROONOSC(self.df['high'], self.df['low'], timeperiod=7)
        real.name = 'AROONOSC'
        return real
    
    # 24. Long Short term Aroon Oscillator [self-developed]
    def ls_aroonosc(self):
        aroonosc1 = talib.AROONOSC(self.df['high'], self.df['low'], timeperiod=3)
        aroonosc2 = talib.AROONOSC(self.df['high'], self.df['low'], timeperiod=5)
        real = aroonosc1 - aroonosc2
        real.name = 'LS_AROONOSC'
        return real
    
    # 25. Balance Of Power
    def bop(self):
        real = talib.BOP(self.df['open'], self.df['high'], self.df['low'], self.df['close'])
        real.name = 'BOP'
        return real
    
    # 26. Commodity Channel Index
    def cci(self):
        real = talib.CCI(self.df['high'], self.df['low'], self.df['close'], timeperiod=2)
        real.name = 'CCI'
        return real
    
    # 27. Long Short term Commodity Channel Index [self-developed]
    def ls_cci(self):
        cci1 = talib.CCI(self.df['high'], self.df['low'], self.df['close'], timeperiod=2)
        cci2 = talib.CCI(self.df['high'], self.df['low'], self.df['close'], timeperiod=10)
        real = cci1 - cci2
        real.name = 'LS_CCI'
        return real
    
    # 28. Chande Momentum Oscillator
    def cmo(self):
        real = talib.CMO(self.df['close'], timeperiod=3)
        real.name = 'CMO'
        return real
    
    # 29. Long Short term Chande Momentum Oscillator [self-developed]
    def ls_cmo(self):
        cmo1 = talib.CMO(self.df['close'], timeperiod=3)
        cmo2 = talib.CMO(self.df['close'], timeperiod=4)
        real = cmo1 - cmo2
        real.name = 'LS_CMO'
        return real
    
    # 30. Directional Movement Index
    def dx(self):
        real = talib.DX(self.df['high'], self.df['low'], self.df['close'], timeperiod=3)
        real.name = 'DX'
        return real
    
    # 31. Moving Average Convergence/Divergence
    def macd(self):
        macd, macdsignal, macdhist = talib.MACD(self.df['close'], fastperiod=6, slowperiod=15, signalperiod=5)
        real = macdsignal
        real.name = 'MACD'
        return real
    
    # 32. MACD with controllable MA type
    def macd_ext(self):
        macd, macdsignal, macdhist = talib.MACDEXT(
            self.df['close'], 
            fastperiod=6, fastmatype=4, slowperiod=15, 
            slowmatype=2, signalperiod=5, signalmatype=5
        )
        real = macdsignal
        real.name = 'MACD_ext'
        return real
    
    # 33. Moving Average Convergence/Divergence Fix 12/26
    def macd_fix(self):
        macd, macdsignal, macdhist = talib.MACDFIX(self.df['close'], signalperiod=2)
        real = macdsignal
        real.name = 'MACD_FIX'
        return real
    
    # 34. Money Flow Index
    def mfi(self):
        real = talib.MFI(self.df['high'], self.df['low'], self.df['close'], self.df['volume'], timeperiod=6)
        real.name = 'MFI'
        return real
    
    # 35. Long Short term Money Flow Index [self-developed]
    def ls_mfi(self):
        mfi1 = talib.MFI(self.df['high'], self.df['low'], self.df['close'], self.df['volume'], timeperiod=2)
        mfi2 = talib.MFI(self.df['high'], self.df['low'], self.df['close'], self.df['volume'], timeperiod=5)
        real = mfi1 - mfi2
        real.name = 'LS_MFI'
        return real
    
    # 36. Over bought & Sold MFI [self-developed]
    def obs_mfi(self):
        mfi = talib.MFI(self.df['high'], self.df['low'], self.df['close'], self.df['volume'], timeperiod=6)

        def submap(x):
            if x < 10:
                return 1 - x / 40
            elif x > 90:
                return 1 - x / 60
            else:
                return 1 - x / 50

        real = mfi.apply(submap)
        real.name = 'OBS_MFI'
        return real
    
    # 37. Momentum
    def mom(self):
        real = talib.MOM(self.df['close'], timeperiod=2)
        real.name = 'MOM'
        return real
    
    # 38. Long Short term Momentum [self-developed]
    def ls_mom(self):
        mom1 = talib.MOM(self.df['close'], timeperiod=2)
        mom2 = talib.MOM(self.df['close'], timeperiod=8)
        real = mom1 - mom2
        real.name = 'LS_MOM'
        return real
    
    # 39. Percentage Price Oscillator
    def ppo(self):
        real = talib.PPO(self.df['close'], fastperiod=2, slowperiod=3, matype=4)
        real.name = 'PPO'
        return real
    
    # 40. Rate of change : ((price/prevPrice)-1)*100
    def roc(self):
        real = talib.ROC(self.df['close'], timeperiod=2)
        real.name = 'ROC'
        return real
    
    # 41. Relative Strength Index
    def rsi(self):
        real = talib.RSI(self.df['close'], timeperiod=2)
        real.name = 'RSI'
        return real
    
    # 42. Long Short term Relative Strength Index [self-developed]
    def ls_rsi(self):
        rsi1 = talib.RSI(self.df['close'], timeperiod=2)
        rsi2 = talib.RSI(self.df['close'], timeperiod=3)
        real = rsi1 - rsi2
        real.name = 'LS_RSI'
        return real
    # 43. Over bought & Sold Relative Strength Index compromised with Volume [self-developed]
    def obs_rsiv(self):
        rsi = talib.RSI(self.df['close'], timeperiod=2)

        def submap(x):
            if x < 10:
                return 1 - x / 30
            elif x > 90:
                return 1 - x / 70
            else:
                return 1 - x / 50

        real = rsi.apply(submap) * self.df['volume']
        real.name = 'OBS_RSIV'
        return real
    
    # 44. Stochastic
    def stoch(self):
        slowk, slowd = talib.STOCH(
            self.df['high'], self.df['low'], self.df['close'], 
            fastk_period=3, slowk_period=2, slowd_period=2, slowk_matype=3,  slowd_matype=2
        )
        real = slowk - slowd
        real.name = 'STOCH'
        return real

    # 45. Stochastic Fast
    def stochf(self):
        fastk, fastd = talib.STOCHF(
            self.df['high'], self.df['low'], self.df['close'], 
            fastk_period=5, fastd_period=2, fastd_matype=0
        )
        real = fastk - fastd
        real.name = 'STOCHF'
        return real
    
    # 46. Stochastic Relative Strength Index
    def stochrsi(self):
        fastk, fastd = talib.STOCHRSI(self.df['close'], timeperiod=12, fastk_period=60, fastd_period=2, fastd_matype=0)
        real = fastk - fastd
        real.name = 'STOCHRSI'
        return real
    
    # 47. 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    def trix(self):
        real = talib.TRIX(self.df['close'], timeperiod=5)
        real.name = 'TRIX'
        return real
    
    # 48. Ultimate Oscillator
    def ultosc(self):
        real = talib.ULTOSC(self.df['high'], self.df['low'], self.df['close'], timeperiod1=2, timeperiod2=6, timeperiod3=10)
        real.name = 'ULTOSC'
        return real

    # 49. Williams' %R
    def willr(self):
        real = talib.WILLR(self.df['high'], self.df['low'], self.df['close'], timeperiod=2)
        real.name = 'WILLR'
        return real

    # 50. Long Short term Williams' %R  [self-developed]
    def ls_willr(self):
        willr1 = talib.WILLR(self.df['high'], self.df['low'], self.df['close'], timeperiod=2)
        willr2 = talib.WILLR(self.df['high'], self.df['low'], self.df['close'], timeperiod=9)
        real = willr1 - willr2
        real.name = 'LS_WILLR'
        return real
    
    """
    Volume Indicator Functions
    """
    # 51. Chaikin A/D Line
    def ad(self):
        real = talib.AD(self.df['high'], self.df['low'], self.df['close'], self.df['volume'])
        real.name = 'AD'
        return real
    
    # 52. ADOSC - Chaikin A/D Oscillator
    def adosc(self):
        real = talib.ADOSC(self.df['high'], self.df['low'], self.df['close'], self.df['volume'], fastperiod=2, slowperiod=3)
        real.name = 'ADOSC'
        return real
    
    # 53. On Balance Volume
    def obv(self):
        real = talib.OBV(self.df['close'], self.df['volume'])
        real.name = 'OBV'
        return real
    
    """
    Volatility Indicator Functions
    """
    # 54. Average True Range
    def atr(self):
        real = talib.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=5)
        real.name = 'ATR'
        return real

    # 55. Long Short Term ATR [self-developed]
    def ls_atr(self):
        atr1 = talib.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=3)
        atr2 = talib.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=9)
        real = atr1 - atr2
        real.name = 'LS_ATR'
        return real
    
    # 56. Normalized Average True Range
    def natr(self):
        real = talib.NATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=3)
        real.name = 'NATR'
        return real
    
    # 57. Long Short Term NATR [self-developed]
    def ls_natr(self):
        natr1 = talib.NATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=4)
        natr2 = talib.NATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=8)
        real = natr1 - natr2
        real.name = 'LS_NATR'
        return real
    
    # 58. True Range
    def trange(self):
        real = talib.TRANGE(self.df['high'], self.df['low'], self.df['close'])
        real.name = 'TRANGE'
        return real

    """
    Cycle Indicators
    """
    # 59. Hilbert Transform - Dominant Cycle Period
    def ht_dcperiod(self):
        ht_dcperiod = talib.HT_DCPERIOD(self.df['close'])
        real.name = 'HT_DCPERIOD'
        return real
    
    # 60. Hilbert Transform - Dominant Cycle Phase
    def ht_dcphase(self):
        real = talib.HT_DCPHASE(self.df['close'])
        real.name = 'HT_DCPHASE'
        return real

    # 61. Hilbert Transform - Phasor Components
    def ht_phasor(self):
        inphase, quadrature = talib.HT_PHASOR(self.df['close'])
        real = inphase + quadrature
        real.name = 'HT_PHASOR'
        return real

    # 62. Hilbert Transform - SineWave
    def ht_sine(self):
        sine, leadsine = talib.HT_SINE(self.df['close'])
        real = sine
        real.name = 'HT_SINE'
        return real

