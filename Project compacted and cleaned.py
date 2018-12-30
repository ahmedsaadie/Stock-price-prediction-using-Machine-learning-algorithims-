from __future__ import print_function

# Global Constants
symbols = ['AAPL','AMZN','MSFT','GOOG','IBM','ORCL','INTC','HPQ','LNVGY','^IXIC']  # add more symbols here. Must have atleast one symbol aside from NASDAQ
START_DATE = '1995-01-01'         # Starting day of scraping
END_DATE = '2018-09-01'           # Final day of scraping
PAST_N_DAYS = 5                   # how many previous days should be used for averaging stock price and movement volatility

# Imports
import numpy as np
import pandas as pd
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from os.path import (dirname,join,)
from pandas_datareader.data import DataReader
from datetime import timedelta, date

def daterange(_start_date, _end_date):
    for n in range(int( (_end_date - _start_date).days) ):
        yield _start_date + timedelta(n)
        
        
#-------------------------------------------------------------------------------#
#----------Scrape stock data and store in SYMBOL.csv----------------------------#
#-------------------------------------------------------------------------------#

# Specifically chosen to include the AAPL split on June 9, 2014.
for symbol in symbols:

    data = DataReader(symbol,'yahoo',start= START_DATE,end= END_DATE,)
    data.rename(columns={'Close': 'close','Volume': 'volume',},inplace=True,)
    del data['Adj Close']
    del data['High']
    del data['Low']
    del data['Open']
    dest = join(symbol + '.csv')
    print("Writing %s -> %s" % (symbol, dest))
    data.to_csv(dest, index_label='Date')

# data2 = data.set_index(close)
for symbol in symbols: 

    file = join(symbol + '.csv')

    xl = pd.read_csv(file)
    date_column = xl['Date']
    Close_column = xl['close']
    Sym = []
    AD_Position = Close_column.copy() #REMOVE??????
    Change = [0]* len(date_column)
    Momentum = [0]* len(date_column)
    file1_Stock_Momentum = [0]* len(date_column)
    file1_Close = [0]* len(date_column)
    dates = [i.split(' ', 1)[0] for i in date_column]
    reference = dates[0]
    
    for i in range (1,len(date_column)):

        if Close_column[i] > Close_column[i-1] :
            Momentum[i] = int("1")
            #Change[i] = (Close_column[i]-Close_column[i-1])/Close_column[i-1]
        else :
            Momentum[i] = int('-1') #int("0")
            #Change[i] = (Close_column[i-1] - Close_column[i]) / Close_column[i - 1]
        
        Change[i] = (Close_column[i]-Close_column[i-1])/Close_column[i-1] # Percentage change
        Sym = symbol
    xl = pd.DataFrame({'Date':date_column, 'Close':Close_column,'Change':Change,'Momentum':Momentum,'Symbol':Sym}) # '5Day Average Change':prev_5_day_avg_change, 'Average 5Day Change in Momentum':prev_5_day_avg_Momentum}) # a represents closing date b represents closing value c represents close change and d represents momentum
    dest = join("modified" + file)
    xl.to_csv(dest,index=True,index_label="Index",header=True)
    print("Writing %s -> %s" % (file, dest))


#-------------------------------------------------------------------------------#
#-------------------Create aggregated Dataset for ML----------------------------#
#-------------------------------------------------------------------------------#


# initialize columns that will be used in the combined dataframe
symbol_date, symbol_close, symbol_change, symbol_momentum, symbol_name, symbol_name, symbol_index = [],[],[],[],[],[],[]

for symbol in symbols:
    symbol_filename = join("modified" + symbol+".csv")
    symbol_csv = pd.read_csv(symbol_filename)
    symbol_date.extend(symbol_csv['Date'])
    symbol_close.extend(symbol_csv['Close'])
    symbol_change.extend(symbol_csv['Change'])
    symbol_momentum.extend(symbol_csv['Momentum'])
    symbol_name.extend(symbol_csv['Symbol'])
    symbol_index.extend(symbol_csv['Index'])
    
dataframe = pd.DataFrame()
dataframe['Date'] = symbol_date
dataframe['Date'] = dataframe['Date'].values.astype('datetime64[D]') # as datetime object
dataframe['Close'] = symbol_close
dataframe['Change'] = symbol_change
dataframe['Momentum'] = symbol_momentum
dataframe['Symbol'] = symbol_name
dataframe['Indices'] = symbol_index

# Create copy dataset of only essential columns that are sorted by date 
Date, Symbol, Close,Change,Stock_Price_Volatility,Stock_Momentum, Index_Volatility,Index_Momentum, Sector_Momentum = [],[],[],[],[],[],[],[],[]

for day in daterange(parse(START_DATE).date(),parse(END_DATE).date()):
    print("Creating dataset row for date: " + str(day) )
    select_daily_index_row = dataframe.loc[(dataframe['Date'] == day) & (dataframe['Indices'] > PAST_N_DAYS ) & (dataframe['Symbol'] == "^IXIC" )]
    for index, row in select_daily_index_row.iterrows():
        N_days_index_volatility = 0.0
        N_days_index_momentum   = 0.0
        count = 0        
        
        # Calculate volatility and momentum for PAST_N_DAYS
        for back_date in range(1,PAST_N_DAYS+1):
            #print(index-back_date)
            N_days_index_volatility += dataframe.iloc[index-back_date]['Change']
            N_days_index_momentum += dataframe.iloc[index-back_date]['Momentum']            
            count = count + 1
        N_days_index_volatility = N_days_index_volatility / count
        N_days_index_momentum = N_days_index_momentum / count
    
    # Get stock prices and their volatility
    df_selection = dataframe.loc[(dataframe['Date'] == day) & (dataframe['Indices'] > PAST_N_DAYS ) & (dataframe['Symbol'] != "^IXIC" )] # Select only data of specified day which has atleast "PAST_N_DAYS" of specified history of Change column
    sector_momentum_sum     = 0
    sector_company_count    = 0
    for index, row in df_selection.iterrows():
        #print(index, row["Symbol"], row["Indices"])        
        N_days_price_volatility = 0.0
        N_days_momentum         = 0.0
        count = 0        
        # Calculate volatility and momentum for PAST_N_DAYS for sector stocks
        for back_date in range(1,PAST_N_DAYS+1):
            #print(index-back_date)
            N_days_price_volatility += dataframe.iloc[index-back_date]['Change']
            N_days_momentum += dataframe.iloc[index-back_date]['Momentum']            
            count = count + 1
        N_days_price_volatility = N_days_price_volatility / count
        N_days_momentum = N_days_momentum / count
        
        Date.append(row["Date"])
        Symbol.append(row["Symbol"])
        
        Close.append(row["Close"])
        Change.append(row["Momentum"])
        Stock_Price_Volatility.append(N_days_price_volatility)
        Stock_Momentum.append(N_days_momentum)
        Index_Volatility.append(N_days_index_volatility)
        Index_Momentum.append(N_days_index_momentum)
        sector_momentum_sum += N_days_momentum
        sector_company_count = sector_company_count + 1
        
    for index, row in df_selection.iterrows():
        cumulative_sector_momentum = sector_momentum_sum/sector_company_count
        Sector_Momentum.append(cumulative_sector_momentum)
        #print(cumulative_sector_momentum)


xl = pd.DataFrame({'Date':Date, 'Symbol':Symbol, 'Close':Close, 'Change':Change,'Stock_Price_Volatility':Stock_Price_Volatility,'Stock_Momentum':Stock_Momentum,'Index_Volatility':Index_Volatility,'Index_Momentum':Index_Momentum,'Sector_Momentum':Sector_Momentum}) # a represents closing date b represents closing value c represents close change and d represents momentum
xl.to_csv("Input_Dataset_v2.csv",index=False,header=False)

#-------------------------------------------------------------------------------#
#-------------------------------Perform ML--------------------------------------#
#-------------------------------------------------------------------------------#

df = xl
X = df.values[:,4:9]
Y = df.values[:,3]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
print("Performing Logistic Regression")
LR_Model = LogisticRegression()
LR_Model.fit(X_train, y_train)
LR_Model.score(X_train, y_train)
#print('Coefficient: \n', LR_Model.coef_)
#print('Intercept: \n', LR_Model.intercept_)
y_Logistic = LR_Model.predict(X_test)
#print ("LR accuracy is ", accuracy_score(y_test,y_Logistic)*100)

# SUPPORT VECTOR MACHINES
from sklearn.svm import SVC
print("Performing Support Vector Machines")
SV_Model = SVC(random_state=0)
SV_Model.fit(X_train, y_train)
SV_Model.score(X_train, y_train)
y_SV = SV_Model.predict(X_test)
#print ("SVM accuracy is ", accuracy_score(y_test,y_SV)*100)

# RANDOM FORESTS
from sklearn.ensemble import RandomForestClassifier
print("Performing Random Forest")
RF_Model= RandomForestClassifier()
RF_Model.fit(X_train, y_train)
RF_Model.score(X_train, y_train) 
y_RF = RF_Model.predict(X_test)
#print ("Random Forest accuracy is ", accuracy_score(y_test,y_RF)*100)

#-------------------------------------------------------------------------------#
#--------------------------ACCURACY OUTPUT--------------------------------------#
#-------------------------------------------------------------------------------#


print ("LR accuracy is ", accuracy_score(y_test,y_Logistic)*100)
print ("SVM accuracy is ", accuracy_score(y_test,y_SV)*100)
print ("Random Forest accuracy is ", accuracy_score(y_test,y_RF)*100)



#-------------------------------------------------------------------------------#
#--------------------------REAL LIFE EXAMPLE------------------------------------#
#-------------------------------------------------------------------------------#

import random
accuracy_rate = 0.58
N_transactions = 2000
cost_of_trading = 10     # $10 per trade
avg_sector_stock_price = 87     # Acquired from averaging all stock prices within a sector from 1995-2018. Lets say, a stock bought costs this amount
sector_movement_trend =  0.000959657  # Acquired from averaging all stock price's volatility within a sector from 1995-2018. + means upward movement and minus means downward movement
winloss_payout_per_stock = avg_sector_stock_price * sector_movement_trend # Potential money won/lost per stock if the stock movement was guessed correctly/incorrectly
ROI_per_stock = winloss_payout_per_stock / avg_sector_stock_price # money made/lost per stock if guessed movement correctly/incorrectly


profit_list = []
ROI_list = []
profit_loss = 0.0
ROI_per_n_transactions  = 0
for profit_loss_amount in range(10,1000,10):
    initial_investment =  profit_loss_amount / ROI_per_stock
    profit_loss = 0
    for i in range(N_transactions):        
        x = random.random()
        if (x<=0.58):
            profit_loss += profit_loss_amount - cost_of_trading
        else:
            profit_loss += (-1 * profit_loss_amount) - cost_of_trading
    profit_list.append((profit_loss))
    ROI_per_n_transactions = (profit_loss / initial_investment) *100
    ROI_list.append(ROI_per_n_transactions)
    print("After "+ str(N_transactions)+ " transactions with initial investment of $" + str(int(initial_investment)) + ", ROI will be " + str(round(ROI_per_n_transactions,2)) + "%" )
        


