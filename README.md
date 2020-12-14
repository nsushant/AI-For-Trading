# AI-For-Trading
These are my projects from the Udacity AI For Trading Nanodegree program. The projects are listed on the Udacity github page and resources used can be found there.
Over 10 projects areas covered include an introduction to the world of Quantitative Trading and the building, testing and optimization of trading strategies. 

1.Trading with momentum: 
The momentum trading strategy is useful when dealing with low volitility trading. This trading strategy is implemented using data from Quotemedia.
This trading strategy has rules, buy when the price is rising and sell when it has peaked. 

2.Breakout strategy:
The breakout trading strategy is useful when dealing with high volitility trading.A certain tolerance is placed and above this tolerance the stock is sold.
If below the given threshold, the stock is bought. 

3.Smart Beta And Portfolio Optimization:
In this project, a smart beta portfolio is built and compared to a benchmark index. 
To find out how well the smart beta portfolio did, the tracking error is calculated against the index. 
Then a portfolio by using quadratic programming to optimize the weights. 
This portfolio is rebalanced and turn over is calculated to evaluate the performance. 
this metric is used to find the optimal rebalancing Frequency. 
For the dataset, the end of day from Quotemedia is used.

4.Alpha research And Factor Modelling:
In this project, a statistical risk model is built using PCA. 
This model is used to build a portfolio along with 5 alpha factors. 
these factors are then created, then evaluated using factor-weighted returns, quantile analysis, sharpe ratio, and turnover analysis. 
At the end of the project,  portfolio is optimized using the risk model and factors using multiple optimization formulations. 
For the dataset, the end of day from Quotemedia and sector data from Sharadar is used.

5.NLP On Financial Statements:
In this project, NLP Analysis is performed on 10-k financial statements to generate an alpha factor. 
For the dataset, we'll be using the end of day from Quotemedia and Loughran-McDonald sentiment word lists.

6.Sentiment Analysis with Neural Networks.:
In this project, a deep learning model is built to classify the sentiment of messages from StockTwits, a social network for investors and traders. 
the model will be able to predict if any particular message is positive or negative. 
From this, we'll be able to generate a signal of the public sentiment for various ticker symbols.

7.Combining Signals for Enhanced Alpha:
In this project, signals on a random forest are combined for enhanced alpha. 
While implementing this, the problem of overlapping samples is encountered and solved. 
For the dataset, we'll be using the end of day from Quotemedia and sector data from Sharadar.

8.Backtesting
A backtester is built using barra data. 

Reference:
Udacity project discriptions
