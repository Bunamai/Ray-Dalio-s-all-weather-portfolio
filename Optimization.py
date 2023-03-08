import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime as dt

# Define a class which draws stock/etf data and calculate the return
class Stock:
    def __init__(self, name, start_date, end_date):
        self.data = yf.download(name, start=start_date, end=end_date)
        self.weights = np.arange(0, 1.001, 0.001)
        self.data["Return"] = np.log(self.data["Close"] / self.data["Close"].shift())
        self.data.dropna(inplace=True)
        self.annualized_return = np.expm1(252 * self.data["Return"].mean())

    def get_weights(self):
        return self.weights

    def get_return(self):
        return self.data["Return"]

    def get_annualized_return(self):
        return self.annualized_return

# Initialize stocks in Ray Dalio all weather portfolio
start_date = "2010-01-01"
end_date = dt.today()

TLT = Stock("TLT", start_date, end_date)
IEF = Stock("IEF", start_date, end_date)
GLD = Stock("GLD", start_date, end_date)
DBC = Stock("DBC", start_date, end_date)
VTI = Stock("VTI", start_date, end_date)

Stocks_returns = []
Stocks_annualized_return = []
for stock in [TLT, IEF, GLD, DBC, VTI]:
    Stocks_returns.append(stock.get_return())
    Stocks_annualized_return.append(stock.get_annualized_return())

Returns = pd.DataFrame()
for TLT_weight in np.arange(0, 1.001, 0.01):
    for IEF_weight in np.arange(0, 1.001, 0.01):
        for GLD_weight in np.arange(0, 1.001, 0.01):
            for DBC_weight in np.arange(0, 1.001, 0.01):
                for VTI_weight in np.arange(0, 1.001, 0.01):
                    weights = [TLT_weight, IEF_weight, GLD_weight, DBC_weight, VTI_weight]
                    if np.sum(weights) == 1:
                        print(weights)
                        Port_return = np.matmul(Stocks_annualized_return, np.transpose(weights))
                        Port_std = np.sqrt(252 * np.matmul(np.matmul(weights, np.cov(Stocks_returns)), np.transpose(weights)))
                        Port_Sharpe = (Port_return - 0.04) / Port_std
                        Return = pd.DataFrame({"Expected Return": Port_return, "Standard Deviation": Port_std \
                                               ,"Sharpe Ratio": Port_Sharpe, "TLT weight": TLT_weight, "IEF weight": IEF_weight \
                                               , "GLD weight": GLD_weight, "DBC weight": DBC_weight, "VTI weight": VTI_weight}\
                                              , index=[1,])
                        Returns = pd.concat([Returns, Return], ignore_index=True)

Returns.sort_values(by="Sharpe Ratio", ascending=False, inplace=True)
Returns.to_csv("Portfolio Return SR.csv")
Returns.sort_values(by="Standard Deviation", ascending=True, inplace=True)
Returns.to_csv("Portfolio Return SD.csv")

