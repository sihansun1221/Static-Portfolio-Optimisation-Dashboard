# Static-Portfolio-Optimisation-Dashboard

The project developed a dashboard tailored for investors with limited investment expertise who have already identified up to five stocks they wish to invest in. The dashboard retrieves historical stock price data by scraping tickers from Yahoo.finance and uses the Mean-Variance Optimization (MVO) approach to forecast future prices and optimize the portfolio according to the investor's risk preferences. This functionality provides investors with valuable insights into the optimal allocation of their investments across the selected stocks. The dashboard is designed to be user-friendly, featuring various interactive elements such as input fields, sliders, and visualizations, including data tables and charts.

The code for the project is organized into four PDFs, each focusing on a specific aspect of the implementation:

1. The first part includes the code of needed libraries importing and the process of downloading and preparing stock data from Yahoo.finance
2. The second part includes the code of the methodology we used for forecasting future stock prices - ARIMA model and to optimize a stock portfolio based on MVO (Mean-Variance Optimization)
3. The third part includes the code for creating a dash application that creates an interactive portfolio optimization dashboard
4. The fourth part includes the code for setting up a Dash web application that functions as a "Portfolio Optimization Dashboard", which allows users to input stock tickers, set a risk preference, and see optimized portfolio results based on the inputs. 

Screenshots of the dashboard can be found in the "Dashboard.png" file.
