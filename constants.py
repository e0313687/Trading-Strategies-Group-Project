import os

# Trading days
start_date = '2019-01-01'
end_date = '2022-11-05'

# Project url
project_url = os.path.curdir

# Dir url
data_dir_url = os.path.join(project_url, 'data')

# Preprocessed CSV File
inst_buy_file = os.path.join(data_dir_url, 'inst_buy.csv')
inst_sell_file = os.path.join(data_dir_url, 'inst_sell.csv')
retail_buy_file = os.path.join(data_dir_url, 'retail_buy.csv')
retail_sell_file = os.path.join(data_dir_url, 'rtl_sell.csv')

# Old ticker to new ticker
updated_ticker = {
    '41A.SI':'8K7.SI', 
    'O32.SI': 'VC2.SI',
    'A68U.SI': 'HMN.SI',
    'CNNU.SI' :'CWBU.SI',
    'CH8.SI': 'QES.SI'
}

LONG = {'desc': 'LONG', 'value': 1}
SHORT = {'desc': 'SHORT', 'value': -1}