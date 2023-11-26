import MetaTrader5 as mt5
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from forex_python.converter import CurrencyRates
from currency_converter import CurrencyConverter

from random import sample

from tqdm import tqdm
import vectorbt as vbt

from pandas.tseries.offsets import BDay
from itertools import combinations, count

import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

from threading import Thread, Lock
from multiprocessing import Process

import requests
import json
import hashlib

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cycler

colors = cycler('color', ['669FEE', '66EE91', '9988DD', 'EECC55', '88BB44', 'FFBBBB'])
plt.rc('figure', facecolor='313233')
plt.rc('axes', facecolor='313233', edgecolor='none', axisbelow=True, grid=True, prop_cycle=colors, labelcolor='gray')
plt.rc('grid', color='474A4A', linestyle='solid')
plt.rc('xtick', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('legend', facecolor='313233', edgecolor='313233')
plt.rc('text')

# Telegram notification keys
chat_id = "6208180231"
api_key = "5981002179:AAFNOzy5CZ7VprDVMjTPLGmam_mavNjjz4U"


def send_telegram_message(message: str, chat_id: str, api_key: str, ):
    responses = {}

    proxies = None
    headers = {'Content-Type': 'application/json',
               'Proxy-Authorization': 'Basic base64'}
    data_dict = {'chat_id': chat_id,
                 'text': message,
                 'parse_mode': 'HTML',
                 'disable_notification': True}
    data = json.dumps(data_dict)
    url = f'https://api.telegram.org/bot{api_key}/sendMessage'

    requests.packages.urllib3.disable_warnings()
    response = requests.post(url,
                             data=data,
                             headers=headers,
                             proxies=proxies,
                             verify=False)
    return response


def get_symbols(location):
    mt5.initialize()

    symbol_list = []
    for symbol in mt5.symbols_get():
        if location in symbol.path:
            symbol_list.append(symbol.name)

    return symbol_list


def get_bars_from_broker(symbol, start_date, end_date, timeframe=mt5.TIMEFRAME_M1):
    mt5.initialize()
    bars = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    df_bars = pd.DataFrame(bars)
    df_bars["time"] = pd.to_datetime(df_bars["time"], unit="s")
    return df_bars


def get_latest_bar(symbol, timeframe=mt5.TIMEFRAME_M1):
    mt5.initialize()
    bar = mt5.copy_rates_from(
        symbol,
        timeframe,
        datetime.now() + timedelta(hours=3),
        1)  # number of bars
    df_bar = pd.DataFrame(bar)
    df_bar["time"] = pd.to_datetime(df_bar["time"], unit="s")
    return df_bar


def new_bar_event_pairs(name_1, name_2, last_bar_1, last_bar_2):
    mt5.initialize()

    while True:

        new_bar_1 = get_latest_bar(name_1)
        new_bar_2 = get_latest_bar(name_2)

        if new_bar_1['time'].iloc[0] != last_bar_1['time'].iloc[0] or new_bar_2['time'].iloc[0] != \
                last_bar_2['time'].iloc[0]:
            print(f'{name_1}, {name_2}: new tick: {datetime.now()}')
            return

        time.sleep(5)


def get_bars(symbol):
    return pd.read_csv(symbol + '.csv')


def find_filling_mode(symbol):
    mt5.initialize()

    for i in range(2):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": mt5.symbol_info(symbol).volume_min,
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(symbol).ask,
            "type_filling": i,
            "type_time": mt5.ORDER_TIME_GTC}

        result = mt5.order_check(request)

        if result.comment == "Done":
            break

    return i


def open_long(name, lot_size):
    mt5.initialize()
    filling_type = find_filling_mode(name)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": name,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY,
        "price": mt5.symbol_info_tick(name).ask,
        "deviation": 10,
        "type_filling": filling_type,
        "type_time ": mt5.ORDER_TIME_GTC
    }

    result = mt5.order_send(request)
    return result.order


def close_long(name, lot_size, order_id):
    mt5.initialize()
    filling_type = find_filling_mode(name)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": name,
        "position": order_id,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(name).bid,
        "deviation": 10,
        "type_filling": filling_type,
        "type_time ": mt5.ORDER_TIME_GTC
    }

    result = mt5.order_send(request)
    time.sleep(2)
    return result


def open_short(name, lot_size):
    mt5.initialize()
    filling_type = find_filling_mode(name)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": name,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(name).bid,
        "deviation": 10,
        "type_filling": filling_type,
        "type_time ": mt5.ORDER_TIME_GTC
    }

    result = mt5.order_send(request)
    return result.order


def close_short(name, lot_size, order_id):
    mt5.initialize()
    filling_type = find_filling_mode(name)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": name,
        "position": order_id,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY,  # mt5.ORDER_TYPE_SELL
        "price": mt5.symbol_info_tick(name).ask,  # mt5.symbol_info_tick(symbol).bid
        "deviation": 10,
        "type_filling": filling_type,
        "type_time ": mt5.ORDER_TIME_GTC
    }

    result = mt5.order_send(request)  # Sends order to broker
    time.sleep(2)
    return result


def draw_graph(name_1, name_2, data, tp, u_t, l_t, uu_t, ll_t):
    plt.figure().set_figwidth(20)
    plt.title(name_1 + ' - ' + name_2)
    plt.plot(data)

    plt.axhline(tp, color='g')
    plt.axhline(u_t, color='y')
    plt.axhline(l_t, color='y')
    plt.axhline(uu_t, color='r')
    plt.axhline(ll_t, color='r')

    plt.show()


def get_ticket(name, position_type):
    for position in mt5.positions_get():
        if position.symbol == name and position.type == position_type:
            return position.ticket

    return


def calc_position_size(name, risk):
    balance = mt5.account_info().balance

    contract_cost = get_pip_value(name, 'EUR') * mt5.symbols_get(name)[0].trade_contract_size / 30
    lot_size = balance * risk / contract_cost

    return round(lot_size, 2)


def get_pip_value(symbol, account_currency):
    symbol_1 = symbol[0:3]
    symbol_2 = symbol[3:6]

    c = CurrencyConverter()
    return c.convert(c.convert(1, symbol_1, symbol_2), symbol_2, account_currency)


def first_cointegration_test(pair_close_data):
    COINTEGRATION_CONFIDENCE_LEVEL = 95
    result = coint_johansen(pair_close_data, 0, 1)

    confidence_level_cols = {
        90: 0,
        95: 1,
        99: 2
    }
    confidence_level_col = confidence_level_cols[COINTEGRATION_CONFIDENCE_LEVEL]

    trace_crit_value = result.cvt[:, confidence_level_col]
    eigen_crit_value = result.cvm[:, confidence_level_col]

    if np.all(result.lr1 >= trace_crit_value) and np.all(result.lr2 >= eigen_crit_value):
        return True
    else:
        return False


def second_cointegration_test(pair_close_data):
    # Step 1: Perform OLS regression of one time series on the other
    y = pair_close_data.iloc[:, 0]
    x = sm.add_constant(pair_close_data.iloc[:, 1])
    model = sm.OLS(y, x)
    results = model.fit()

    # Step 2: Test residuals for stationarity using ADF test
    residuals = y - results.predict(x)
    adf_result = adfuller(residuals)

    # Step 3: If residuals are stationary, the two time series are considered cointegrated
    if adf_result[1] < 0.05:
        return results.params[1]
    else:
        return None


def compute_Hc(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""

    time_series = time_series.to_numpy()
    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]


def first_sieve(all_pairs_close_data):
    candidate_stationary_pairs_close_data = {}

    for pair, close_data in tqdm(all_pairs_close_data.items()):

        if close_data.empty:
            continue
        else:
            if first_cointegration_test(close_data):
                candidate_stationary_pairs_close_data.update({
                    pair: close_data
                })

    return candidate_stationary_pairs_close_data


def second_sieve(candidate_stationary_pairs_close_data):
    stationary_pairs_close_data = []
    for pair, close_data in tqdm(candidate_stationary_pairs_close_data.items()):
        if close_data.empty:
            continue
        else:
            slope = second_cointegration_test(close_data)
            if slope:
                stationary_pairs_close_data.append({
                    'coint_pair': pair, 'close_data': close_data, 'slope': slope
                })

    return stationary_pairs_close_data


def third_seive(stationary_pairs_close_data):
    consistent_stationary_pairs_close_data = []

    for data in tqdm(stationary_pairs_close_data):
        sym_0 = data['coint_pair'][0]
        sym_1 = data['coint_pair'][1]

        close0 = data['close_data']['close0']
        close1 = data['close_data']['close1']

        slope = data['slope']
        artificial_asset = close0 - slope * close1

        if compute_Hc(artificial_asset) < 0.5:
            consistent_stationary_pairs_close_data.append(data)

    return consistent_stationary_pairs_close_data


def backtest(name_1, name_2, data_close):

    # Positions are encoded as follows
    HIGH = 1
    FREE = 0
    LOW = -1

    mean = data_close.mean()
    std = data_close.std()

    tp = mean
    open_high = mean + (2 * std)
    open_low = mean - (2 * std)

    sl_high = mean + (3 * std)
    sl_low = mean - (3 * std)



    # draw_graph(name_1 + " " + name_2, data_close, tp, open_high, open_low, sl_high, sl_low)

    profit = 0

    entries = [False for i in range(len(data_close))]
    exits = [False for i in range(len(data_close))]

    position_type = FREE

    for t in range(1, len(data_close)):
        # Trade!
        if position_type == FREE:
            if data_close.iloc[t - 1] < open_high <= data_close.iloc[t]:
                # Short first Long last
                entries[t] = True
                position_type = HIGH

            elif data_close.iloc[-2] > open_low >= data_close.iloc[t]:
                # Long first Short last
                entries[t] = True
                position_type = LOW

        elif position_type == HIGH:
            if data_close.iloc[t - 1] > tp >= data_close.iloc[t]:
                exits[t] = True
                position_type = FREE
            elif data_close.iloc[t - 1] < sl_high <= data_close.iloc[t]:
                exits[t] = True
                position_type = FREE
        elif position_type == LOW:
            if data_close.iloc[t - 1] < tp <= data_close.iloc[t]:
                exits[t] = True
                position_type = FREE
            elif data_close.iloc[t - 1] > sl_low >= data_close.iloc[t]:
                exits[t] = True
                position_type = FREE

    return entries, exits


def trade_engine(data, name_1, name_2, lot_size_1, lot_size_2, position_type, long_ticket, short_ticket):

    # Positions are encoded as follows
    HIGH = 1
    FREE = 0
    LOW = -1


    mean = data.mean()
    std = data.std()

    tp = mean
    open_high = mean + (2 * std)
    open_low = mean - (2 * std)

    sl_high = mean + (3 * std)
    sl_low = mean - (3 * std)

    # Trade!
    if position_type == FREE:
        #if data.iloc[-2] < open_high.iloc[-2] and data.iloc[-1] >= open_high.iloc[-1]:
        if data.iloc[-2] < open_high <= data.iloc[-1]:
            # Short first Long last

            send_telegram_message(
                f'{name_1}, {name_2}, Openned High Positions at: \n\n{data.iloc[-1]}, Account Balance is {mt5.account_info().balance}.',
                chat_id, api_key)

            # draw_graph(name_1, name_2, data, tp, open_high, open_low, sl_high, sl_low)
            short_ticket = open_short(name_1, lot_size_1)
            long_ticket = open_long(name_2, lot_size_2)
            position_type = HIGH

        #elif data.iloc[-2] > open_low.iloc[-2] and data.iloc[-1] <= open_low.iloc[-1]:
        elif data.iloc[-2] > open_low >= data.iloc[-1]:
            # Long first Short last

            send_telegram_message(
                f'{name_1}, {name_2}, Openned Low Positions at: \n\n{data.iloc[-1]}, Account Balance is {mt5.account_info().balance}.',
                chat_id, api_key)

            # draw_graph(name_1, name_2, data, tp, open_high, open_low, sl_high, sl_low)
            long_ticket = open_long(name_1, lot_size_1)
            short_ticket = open_short(name_2, lot_size_2)
            position_type = LOW

    elif position_type == HIGH:
        #if data.iloc[-2] > tp.iloc[-2] and data.iloc[-1] <= tp.iloc[-1]:
        if data.iloc[-2] > tp >= data.iloc[-1]:

            send_telegram_message(
                f'{name_1}, {name_2}, TP High Positions at: \n\n{data.iloc[-1]}, Account Balance is {mt5.account_info().balance}.',
                chat_id, api_key)
            # draw_graph(name_1, name_2, data, tp, open_high, open_low, sl_high, sl_low)

            close_short(name_1, lot_size_1, short_ticket)
            close_long(name_2, lot_size_2, long_ticket)
            short_ticket = 0
            long_ticket = 0
            position_type = FREE

        #elif data.iloc[-2] < sl_high.iloc[-2] and data.iloc[-1] >= sl_high.iloc[-1]:
        elif data.iloc[-2] < sl_high <= data.iloc[-1]:

            send_telegram_message(
                f'{name_1}, {name_2}, SL High Positions at: \n\n{data.iloc[-1]}, Account Balance is {mt5.account_info().balance}.',
                chat_id, api_key)
            # draw_graph(name_1, name_2, data, tp, open_high, open_low, sl_high, sl_low)

            close_short(name_1, lot_size_1, short_ticket)
            close_long(name_2, lot_size_2, long_ticket)
            short_ticket = 0
            long_ticket = 0

            position_type = FREE

    elif position_type == LOW:
        #if data.iloc[-2] < tp.iloc[-2] and data.iloc[-1] >= tp.iloc[-1]:
        if data.iloc[-2] < tp <= data.iloc[-1]:

            send_telegram_message(
                f'{name_1}, {name_2}, TP Low Positions at: \n\n{data.iloc[-1]}, Account Balance is {mt5.account_info().balance}.',
                chat_id, api_key)
            # draw_graph(name_1, name_2, data, tp, open_high, open_low, sl_high, sl_low)

            close_long(name_1, lot_size_1, long_ticket)
            close_short(name_2, lot_size_2, short_ticket)
            long_ticket = 0
            short_ticket = 0
            position_type = FREE

        #elif data.iloc[-2] > sl_low.iloc[-2] and data.iloc[-1] <= sl_low.iloc[-1]:
        elif data.iloc[-2] > sl_low >= data.iloc[-1]:

            send_telegram_message(
                f'{name_1}, {name_2}, SL Low Positions at: \n\n{data.iloc[-1]}, Account Balance is {mt5.account_info().balance}.',
                chat_id, api_key)
            # draw_graph(name_1, name_2, data, tp, open_high, open_low, sl_high, sl_low)

            close_long(name_1, lot_size_1, long_ticket)
            close_short(name_2, lot_size_2, short_ticket)
            long_ticket = 0
            short_ticket = 0
            position_type = FREE

    return position_type, long_ticket, short_ticket


def pairs_trader(pair, risk):

    name_1 = pair[0]['coint_pair'][0]
    name_2 = pair[0]['coint_pair'][1]
    slope = pair[0]['slope']

    data_1_close = pair[0]['close_data']['close0']
    data_2_close = pair[0]['close_data']['close1']

    artificial_asset = data_1_close + slope * data_2_close

    position_type = 0
    long_ticket = 0
    short_ticket = 0

    print(f'Process: {name_1}, {name_2} started.')

    draw_graph(
        name_1,
        name_2,
        artificial_asset,
        artificial_asset.mean(),
        artificial_asset.mean() + 2 * artificial_asset.std(),
        artificial_asset.mean() - 2 * artificial_asset.std(),
        artificial_asset.mean() + 3 * artificial_asset.std(),
        artificial_asset.mean() - 3 * artificial_asset.std()
    )

    ##################################################
    # Trade Loop

    while True:
        last_bar_1 = get_latest_bar(name_1)
        last_bar_2 = get_latest_bar(name_2)

        new_bar_event_pairs(name_1, name_2, last_bar_1, last_bar_2)

        end_date = datetime.now() + timedelta(hours=3)
        start_date = end_date - BDay(10)

        close_1 = get_bars_from_broker(name_1, start_date, end_date)['close']
        close_2 = get_bars_from_broker(name_2, start_date, end_date)['close']

        aa_data = close_1 + slope * close_2

        lot_size_1 = calc_position_size(name_1, risk=risk)
        lot_size_2 = calc_position_size(name_2, risk=risk)

        position_type, long_ticket, short_ticket = trade_engine(
            aa_data,
            name_1,
            name_2,
            lot_size_1,
            lot_size_2,
            position_type,
            long_ticket,
            short_ticket
        )

    ##################################################


if __name__ == '__main__':

    mt5.initialize()

    end_date = datetime.now() + timedelta(hours=3)
    start_date = end_date - BDay(10)

    # 1 Get Symbol List
    location = "Retail\\Forex"

    symbol_list = []
    for symbol in mt5.symbols_get():
        if location in symbol.path:
            if 'CNH' in symbol or 'RUB' in symbol or 'TRY' in symbol:
                continue
            symbol_list.append(symbol.name)

    symbol_close_data = {}
    for symbol in tqdm(symbol_list):
        bar_data = get_bars_from_broker(symbol, start_date, end_date)

        symbol_close_data.update({
            symbol: bar_data['close']
        })

        bar_data.to_csv('historic_data/' + symbol + '.csv', index=False)

    symbol_pairs = list(combinations(symbol_list, 2))

    all_pairs_close_data = {}

    for pair in tqdm(symbol_pairs):
        close0 = symbol_close_data[pair[0]]
        close1 = symbol_close_data[pair[1]]
        all_pairs_close_data.update({
            pair: pd.concat([close0, close1], axis=1, keys=['close0', 'close1']).dropna()
        })

    print(len(all_pairs_close_data))

    candidate_stationary_pairs_close_data = first_sieve(all_pairs_close_data)
    print(len(candidate_stationary_pairs_close_data))

    # candidate_stationary_pairs_close_data = all_pairs_close_data
    stationary_pairs_close_data = second_sieve(candidate_stationary_pairs_close_data)
    print(len(stationary_pairs_close_data))

    consistent_pairs_close_data = third_seive(stationary_pairs_close_data)
    print(len(consistent_pairs_close_data))

    profitable_pairs = []
    for data in tqdm(consistent_pairs_close_data):
        name_1 = data['coint_pair'][0]
        name_2 = data['coint_pair'][1]
        slope = data['slope']

        data_1_close = data['close_data']['close0']
        data_2_close = data['close_data']['close1']

        data_close = data_1_close + slope * data_2_close

        entries, exits = backtest(name_1, name_2, data_close)

        try:
            pf = vbt.Portfolio.from_signals(data_close, entries=entries, exits=exits, freq='15m')
        except ValueError:
            continue

        if 1 < pf.sharpe_ratio() < math.inf:

            spread_1 = mt5.symbols_get(name_1)[0].ask - mt5.symbols_get(name_1)[0].bid
            spread_2 = mt5.symbols_get(name_2)[0].ask - mt5.symbols_get(name_2)[0].bid

            risk_adjusted_spread_1 = calc_position_size(name_1, risk=1) * spread_1
            risk_adjusted_spread_2 = calc_position_size(name_2, risk=1) * spread_2

            if risk_adjusted_spread_1 > risk_adjusted_spread_2:
                if risk_adjusted_spread_1 < 2 * risk_adjusted_spread_2:
                    profitable_pairs.append((data, pf.sharpe_ratio(), pf.total_profit()))
            elif risk_adjusted_spread_2 > risk_adjusted_spread_1:
                if risk_adjusted_spread_2 < 2 * risk_adjusted_spread_1:
                    profitable_pairs.append((data, pf.sharpe_ratio(), pf.total_profit()))

    for pair in profitable_pairs:
        print(f'{pair[0]["coint_pair"][0]}, \t\t{pair[0]["coint_pair"][1]}, \tSR: {pair[1]}, \t TP: {pair[2]}')

    risk = round((1 / len(profitable_pairs)) / 2, 3)

    print(f'RISK: {risk}')

    for pair in profitable_pairs:
        name_1 = pair[0]['coint_pair'][0]
        name_2 = pair[0]['coint_pair'][1]
        slope = pair[0]['slope']

        data_1_close = pair[0]['close_data']['close0']
        data_2_close = pair[0]['close_data']['close1']

        artificial_asset = data_1_close + slope * data_2_close

        draw_graph(
            name_1,
            name_2,
            artificial_asset,
            artificial_asset.mean(),
            artificial_asset.mean() + 2 * artificial_asset.std(),
            artificial_asset.mean() - 2 * artificial_asset.std(),
            artificial_asset.mean() + 3 * artificial_asset.std(),
            artificial_asset.mean() - 3 * artificial_asset.std()
        )

        trader = Process(target=pairs_trader, args=(pair, risk))
        trader.start()
        time.sleep(2)
