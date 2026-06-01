import yfinance as yf
import numpy as np
import pandas as pd
import json
from datetime import datetime,date
import re


def get_portfolio_from_input():   
#Take user input, but make sure it is an int for the loop
#Let it loop back if user inputs incorrectly, to avoid restarting everything
    while True:
        start_date=input(f"Enter start date (YYYY-MM-DD): ")
        end_date=input(f"Enter end date (YYYY-MM-DD): ")
        if re.match('^[0-9]{4}-[0-9]{2}-[0-9]{2}$',start_date and end_date) and start_date<end_date:
            break  
        else:
            print("Please make sure that you formatted everything correctly")
            continue  
    
    #Collect raw input data, only save valid tickers to tickers list
    raw_tickers=[item.strip() for item in input("Enter each ticker separated by commas: ").upper().split(',')]
    tickers=[]
    for t in raw_tickers:
        while True:  
            #Check to make sure that ticker is valid
            if yf.Ticker(t).info.get('shortName') is not None:
                tickers.append(t)
                print(f"Ticker found for {t}.")
                break
            else:
                print(f'Ticker not found for {t}')
                t=input("Re-enter a valid ticker to replace it (or press Enter to skip):").upper().strip()
                if t =='':
                    break
        #Check to see if there are any duplicate tickers
        if len(tickers)!=len(set(tickers)):
                print(f'{t} already found in portfolio. Removing {t}.')
                tickers=list(set(tickers))
    return tickers,start_date,end_date,'input',None

def get_portfolio_from_csv():
    #Open file if that is what they choose
    portfolio_name=input("Please enter the file name of your portfolio, including the extension: ")
    if 'csv' in portfolio_name:
        csv_portfolio=pd.read_csv(portfolio_name,parse_dates=['date'])
        tickers=csv_portfolio['ticker'].unique().tolist()
        start_date=csv_portfolio['date'].min()
        end_date=csv_portfolio['date'].max()   
        return tickers,start_date,end_date,'csv',csv_portfolio
    else:
        print('Please make sure you are inputting a csv file')
        return get_portfolio_from_csv()
    

def get_portfolio_from_json():
    portfolio_name=input('Please enter the file name of your portfolio, including the extension: ')
    if 'json' in portfolio_name:
        with open(portfolio_name,"r") as f:
            json_portfolio=json.load(f)
        trades=pd.DataFrame(json_portfolio['portfolio']['trades'])
        start_date=pd.to_datetime(trades['date']).min()
        end_date=pd.to_datetime(trades['date']).max()
        tickers=trades['ticker'].unique().tolist()
        return tickers, start_date, end_date,'json',json_portfolio


#Create the portfolio of tickers and dates
def portfolio():
    choice =input("Type 1 for user input of portfolio, 2 for csv input of portfolio, or 3 for json input of portfolio: ")

    if choice=='1':
        return get_portfolio_from_input()
    if choice=='2':
        return get_portfolio_from_csv()
    if choice=='3':
        return get_portfolio_from_json()
    else:     
        print("Please enter 1,2, or 3")
        return portfolio()


def portfolio_holdings(tickers,source,portfolio_df=None):
    if source=='input':
        while True:
                try:
                    #Create a dictionary with weights of each ticker
                    ticker_holdings={ticker:int(input(f"Enter the number of shares of {ticker} you are holding: ")) for ticker in tickers}
                    pct_weights={ticker:f"{(shares/sum(ticker_holdings.values())):.2%}" for ticker,shares in ticker_holdings.items()}
                    break
                except (ValueError, ZeroDivisionError):
                    print("Please enter an integer number of holdings\n")
        return ticker_holdings,pct_weights,'input'
    elif source=='csv':
        portfolio_df['signed_shares']=portfolio_df.apply(
            lambda row: row['shares'] if row['side']=='buy' else -row['shares'],axis=1
        )
        ticker_holdings=portfolio_df.groupby('ticker')['signed_shares'].sum().to_dict()
        total=sum(ticker_holdings.values())
        pct_weights={ticker:f"{(shares/total):.2%}" for ticker,shares in ticker_holdings.items()}
        return ticker_holdings,pct_weights,'csv'
    elif source=='json':
        trades=pd.DataFrame(portfolio_df['portfolio']['trades'])
        trades['signed_shares']=trades.apply(
            lambda trades: trades['shares'] if trades['side']=='buy' else -trades['shares'],axis=1
        )
        ticker_holdings=trades.groupby('ticker')['signed_shares'].sum().to_dict()
        total=sum(ticker_holdings.values())
        pct_weights={ticker:f"{(shares/total):.2%}" for ticker,shares in ticker_holdings.items()}
        return ticker_holdings,pct_weights,'json'

def fetch_clean_prices(tickers,start,end):
    raw=yf.download(tickers=tickers,start=start,end=end,auto_adjust=True)
    
    #Get log returns and prices into dataframes
    #Returns for if only a single ticker
    if isinstance(tickers,str) or len(tickers)==1:
        prices=raw['Close'].squeeze().to_frame()
        simple_returns=(raw['Close']/raw['Close'].shift(1)).squeeze().to_frame().dropna()
        print(raw['Close'])
        log_returns=np.log(raw['Close']/raw['Close'].shift(1)).dropna().squeeze().to_frame()
        prices.columns=[tickers] if isinstance(tickers,str) else tickers
        log_returns.columns=[tickers] if isinstance(tickers,str) else tickers
    #Returns for more than one ticker
    else:
        prices=raw['Close']
        simple_returns=pd.DataFrame(raw['Close']/raw['Close'].shift(1)).dropna()
        log_returns=pd.DataFrame(np.log(raw['Close']/raw['Close'].shift(1))).dropna()

    #Report any tickers with excessive missing data
    missing_pct=prices.isnull().mean()
    for ticker,pct in missing_pct.items():
        if pct>0.02:
            print(f"Warning: {ticker} has {pct:.1%} missing values")
    
    #Forward fill any small gaps (e.g. one-day holiday closures)
    #Limit=1 means only fill gaps of exactly one day
    prices=prices.ffill(limit=1)
    prices=prices.dropna()
    
    #Report any tickers with 5+ days of same price (may indicate stale prices)
    for ticker in prices.columns:
        rolling_unique=prices[ticker].rolling(5).apply(lambda x:len(set(x)))
        stale_days=(rolling_unique==1).sum()
        if stale_days>0:
            print(f"Warning: {ticker} has {stale_days} potential stale days")
        
    #Report any tickers with price movements >20%
    for ticker in prices.columns:
        spikes=(log_returns[ticker].abs()>0.20).sum()
        if spikes>0:
            print(f"Warning: {ticker} has {spikes} daily moves exceeding 20%")

    print(f"Loaded {len(prices)} trading days for {list(prices.columns)}")
    return prices,log_returns,simple_returns

#Compute daily returns for entire portfolio as weighted sum of individual returns
def daily_returns(pct_weights):
    weighted_returns=pct_weights.sum(axis=1)
    print(weighted_returns)
    return weighted_returns

#Compute different return/vol metrics
def all_metrics(log_returns,rf_annual=0.0525,):
    metrics={}
    for ticker in log_returns.columns:
        returns=log_returns[ticker]
        rf_daily=rf_annual/252
        ann_return=returns.mean()*252
        ann_vol=(returns.std())*np.sqrt(252)
        sharpe=(returns.mean()-rf_daily)/returns.std()*np.sqrt(252)
        #Cummulative metrics
        cummulative=np.exp(returns.cumsum())
        peak=cummulative.cummax()
        max_dd=((cummulative-peak)/peak).min()
        metrics[ticker]={
            'ann_return':float(ann_return),
            'ann_vol':float(ann_vol),
            'sharpe':float(sharpe),
            'max_dd':float(max_dd)
        }
        

    max_vol=max(metrics,key=lambda t:metrics[t]['ann_vol'])
    print(f"Most volatile: {max_vol} at {metrics[max_vol]['ann_vol']:.2%}")
    return metrics,max_vol


#Correlation matrix
def corr_matrix(log_returns):
    corr_matrix=log_returns.corr()
    #Make sure there are no names in columns before assigning names
    corr_matrix.columns.name=None
    corr_matrix.index.name=None
    #Form pairs of ticker_A,ticker_B
    if len(corr_matrix)>1:
        pairs=(corr_matrix.unstack().reset_index().rename(columns={'level_0':'ticker_A','level_1':'ticker_B',0:'correlation'}))
        pairs=pairs[pairs['ticker_A']!=pairs['ticker_B']]
        pairs_sorted=pairs.sort_values('correlation',ascending=False)
        print("\nThe most correlated pair was:")
        print(pairs_sorted.head(1).to_string(index=False))
    else:
        print(f"Only one stock: {log_returns.columns[0]}. Cannot form correlation matrix.")


    

#Define parameters
tickers, start_date, end_date,source,df = portfolio()
prices,log_returns,simple_returns=fetch_clean_prices(tickers,start_date,end_date)
metrics,max_vol=all_metrics(log_returns)
ticker_holdings,pct_weights,holding_source=portfolio_holdings(tickers,source)

#Print Statements
print(simple_returns)
print(log_returns)
#print(json.dumps(metrics,indent=4))


















# ── STEP 3: PORTFOLIO CONSTRUCTION ───────────────────────────────
# Define weights for each ticker (must sum to 1.0)
# Compute daily portfolio returns as weighted sum of individual returns
# Hint: (log_returns * weights).sum(axis=1)


# ── STEP 5: COMPARISON TABLE ─────────────────────────────────────
# Print a formatted table showing each ticker vs the portfolio
# Columns: ticker, annual return, volatility, sharpe, max drawdown
# Hint: f-strings with fixed width formatting make this look clean
# Add a final row for the portfolio itself

# ── STEP 6: VISUALIZATION ────────────────────────────────────────
# Plot 1: cumulative returns for each ticker + portfolio on one chart
# Plot 2: correlation heatmap using matplotlib imshow
# Plot 3: bar chart of individual Sharpe ratios
# Hint: plt.subplot() lets you put all three in one figure 

