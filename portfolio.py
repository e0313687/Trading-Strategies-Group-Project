# EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates

# Data source
import yfinance as yf

# User defined
import constants

class BacktestPortfolio:
    '''
    How to use?
    1. Generate the trade info to have the weekly return [gen_trade_list_info](i.e. input stock for selection later)
    2. Identify a trading strategy and create a 'position' column
    3. Calculate the port return with [calc_port_ret]

    Starting fund is $100,000
    '''

    def __init__(self, port_name, port_funds=1e5):
        self.port_funds = port_funds
        self.port_rets = 0
        self.port_name = port_name
        self.__initial_capital = port_funds

        self.__weekly_summary_trade = {
            'date': [],
            'profit_loss': [],
            'return': []
        }
        self.weekly_summary_hist_df = None
        self.RISK_FREE_RATE = None

    # below agg method for get weekly_return
    def pct_change(self, series: pd.Series):
        return (series.iloc[-1] - series.iloc[0]) / series.iloc[0]

    def entry_date(self, series: pd.Series):
        return series.index[0].strftime('%Y-%m-%d')

    def entry_price(self, series: pd.Series):
        return series.iloc[0]

    def exit_date(self, series: pd.Series):
        return series.index[-1].strftime('%Y-%m-%d')

    def exit_price(self, series: pd.Series):
        return series.iloc[-1]

    def __get_weekly_return(self, df: pd.DataFrame):
        df['date'] = df.index + pd.to_timedelta(-6, unit='d')
        grouped = df.groupby(pd.Grouper(key='date', freq='W-MON')) \
                    .agg([self.pct_change, self.entry_date, self.exit_date, self.entry_price, self.exit_price])
        
        return grouped

    def __gen_trade_info(self, recommend_df: pd.DataFrame) -> pd.DataFrame:
        tickers = recommend_df['stock code'].unique().tolist()

        data = yf.download(tickers, start=constants.start_date, end=constants.end_date)
        data_adj_close = data['Adj Close']
        weekly_return = self.__get_weekly_return(data_adj_close)

        beg_trading_days = recommend_df.sort_index().index.unique()

        weekly_trade_info_list = []

        for beg_dt in beg_trading_days:
            this_week_position = recommend_df.loc[beg_dt]
            this_week_position = this_week_position.reset_index().set_index('stock code')
            this_week_position_tickers = this_week_position.index

            trade_info = this_week_position.copy()

            weekly_return_info = pd.DataFrame(weekly_return.loc[beg_dt, this_week_position_tickers])
            weekly_return_info.columns = ['value']
            weekly_return_info = weekly_return_info.reset_index()

            weekly_return_info_pivoted = weekly_return_info.pivot(index='level_0', 
                                                                  columns='level_1', 
                                                                  values='value')
            trade_info = trade_info.merge(weekly_return_info_pivoted, left_index=True, right_index=True)

            weekly_trade_info_list.append(trade_info)

        trade_info = pd.concat(weekly_trade_info_list)
        trade_info.index.name = 'stock code'
        # trade_info = trade_info.reset_index().drop_duplicates(['published date', 'stock code'])
        trade_info = trade_info.reset_index().set_index('published date')

        return trade_info

    def gen_trade_list_info(self, df_list) -> pd.DataFrame:
        lst = []
        for df in df_list:
            trade_info = self.__gen_trade_info(df)

            lst.append(trade_info)
        
        combined_df = pd.concat(lst)
        combined_df = combined_df.sort_index()

        return combined_df

    def calc_port_ret(self, trade_info: pd.DataFrame, position_col='position') -> pd.DataFrame:
        port_hist = []
        trading_days = trade_info.index.unique()

        for trade_date in trading_days:

            weekly_record = trade_info.loc[trade_date].copy()
            weekly_tot_net_amt = weekly_record['net amount (S$M)'].abs().sum()

            # Info: Qty, Entry tot price, Exit tot price, profit loss, port weights
            weekly_record['port_weight'] = weekly_record['net amount (S$M)'].abs() / weekly_tot_net_amt
            weekly_record['max_allocated_funds'] = self.port_funds * weekly_record['port_weight']
            weekly_record['qty'] = np.floor(weekly_record['max_allocated_funds'].abs() / weekly_record['entry_price'] / 100) * 100
            weekly_record['entry_tot_price'] = weekly_record['entry_price'] * weekly_record['qty']
            weekly_record['exit_tot_price'] = weekly_record['exit_price'] * weekly_record['qty']
            weekly_record['port_pct_change'] = weekly_record[position_col] * weekly_record['pct_change']
            weekly_record['port_profit_loss'] = weekly_record[position_col] * (weekly_record['exit_tot_price'] - weekly_record['entry_tot_price'])
            weekly_record['port_weight'] = weekly_record['entry_tot_price'] / self.port_funds

            # Calc weekly return n pnl
            weekly_tot_port_ret = np.dot(weekly_record['port_pct_change'], weekly_record['port_weight'])
            weekly_tot_profit_loss = weekly_record['port_profit_loss'].sum()

            # Add to weekly history
            self.__weekly_summary_trade['date'].append(trade_date)
            self.__weekly_summary_trade['profit_loss'].append(weekly_tot_profit_loss)
            self.__weekly_summary_trade['return'].append(weekly_tot_port_ret)

            self.port_funds += weekly_tot_profit_loss
            self.port_rets += weekly_tot_port_ret

            port_hist.append(weekly_record)

        return pd.concat(port_hist)

    def get_weekly_summary_trade_hist(self):
        self.__weekly_summary_trade['date'].insert(0, constants.start_date)
        self.__weekly_summary_trade['profit_loss'].insert(0, 0)
        self.__weekly_summary_trade['return'].insert(0, 0)

        df = pd.DataFrame(self.__weekly_summary_trade)
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        
        df['cum_ret'] = (1 + df['return']).cumprod()
        df['port_value'] = self.__initial_capital + df['profit_loss'].cumsum()

        self.weekly_summary_hist_df = df

        return df
    
    def port_performance(self):
        if self.RISK_FREE_RATE is None:
            us_treasury_yield_ticker = '^TNX'
            us_treasury_yield = yf.download(us_treasury_yield_ticker, start=constants.start_date, end=constants.end_date)
            self.RISK_FREE_RATE = (us_treasury_yield['Adj Close'] / 100).mean()

        def maximum_drawdown(net_return):
            cum_return = (1 + net_return).cumprod()
            i = np.argmax(np.maximum.accumulate(cum_return) - cum_return) # end of the period
            j = np.argmax(cum_return[:i]) # start of period
            return (cum_return[i] - cum_return[j])/cum_return[j]

        weekly_return = self.weekly_summary_hist_df['return']
        tot_weeks_per_year = 52

        print (f'--- {self.port_name} ---')
        print(f'Portfolio value: $ {self.port_funds:,.2f}')
        print(f'Portfolio return: {self.port_rets * 100:.2f}%')
        print(f'Averge weekly return: {weekly_return.mean() * 100:.2f}%')
        print(f'Volatility weekly return: {weekly_return.std() * 100:.2f}%')
        print(f'Annualized volatility: {(weekly_return.std() * tot_weeks_per_year) ** 0.5:.2f}')
        print(f'Sharpe Ratio: {(weekly_return.mean() * tot_weeks_per_year - self.RISK_FREE_RATE) / (weekly_return.std() * tot_weeks_per_year  ** 0.5):.2f}')
        print(f'Sharpe Ratio: {(weekly_return.mean() * tot_weeks_per_year - self.RISK_FREE_RATE) / (weekly_return[weekly_return < 0].std() * tot_weeks_per_year ** 0.5):.2f}')
        print(f'Maximum drawdown: {maximum_drawdown(weekly_return):.2f}')

    def plot_base(self, value_col, ylabel, title, x_base=False):
        if self.weekly_summary_hist_df is None:
            self.get_weekly_summary_trade_hist()
        df = self.weekly_summary_hist_df

        fig, ax = plt.subplots(figsize=(20, 12))

        ax.plot(df.index, df[value_col])
        if x_base:
            ax.axhline(0, alpha=0.3, c='gray')

        ax.set_xlim(df.index[0], df.index[-1])
        ax.xaxis.set_major_locator(dates.YearLocator())
        ax.xaxis.set_minor_locator(dates.MonthLocator(interval=2))
        ax.xaxis.set_minor_formatter(dates.DateFormatter("%b"))

        ax.tick_params(axis="x", which="major",rotation=45)
        ax.tick_params(axis="x", which="minor",rotation=45)
        ax.xaxis.grid(True)

        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        plt.show()

    def plot_weekly_return(self):
        self.plot_base('return', 'Return', 'Weekly Return', True)

    def plot_weekly_cum_ret(self):
        self.plot_base('cum_ret', 'Cumulative Return', 'Weekly Cumulative Return')

    def plot_weekly_port_value(self):
        self.plot_base('port_value', 'Portfolio Value', f'{self.port_name} Return ({constants.start_date} to {constants.end_date})')
    
    def __repr__(self) -> str:
        output = f'--- {self.port_name} ---'
        output += f'\nAvailable Funds: $ {self.port_funds:,.2f}'
        output += f'\nRealised Total Returns: {self.port_rets * 100:.2f}%'
        
        return output