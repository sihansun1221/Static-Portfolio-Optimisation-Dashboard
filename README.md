# Static-Portfolio-Optimisation-Dashboard

The project developed a dashboard tailored for investors with limited investment expertise who have already identified up to five stocks they wish to invest in. The dashboard retrieves historical stock price data by scraping tickers from Yahoo.finance and uses the Mean-Variance Optimization (MVO) approach to forecast future prices and optimize the portfolio according to the investor's risk preferences. This functionality provides investors with valuable insights into the optimal allocation of their investments across the selected stocks. The dashboard is designed to be user-friendly, featuring various interactive elements such as input fields, sliders, and visualizations, including data tables and charts.

The code for the project is organized into four PDFs, each focusing on a specific aspect of the implementation:

1. The first part includes the code of needed libraries importing and the process of downloading and preparing stock data from Yahoo.finance
2. The second part includes the code of the methodology we used for forecasting future stock prices - ARIMA model and to optimize a stock portfolio based on MVO (Mean-Variance Optimization)
3. The third part includes the code for creating a dash application that creates an interactive portfolio optimization dashboard
4. The fourth part includes the code for setting up a Dash web application that functions as a "Portfolio Optimization Dashboard", which allows users to input stock tickers, set a risk preference, and see optimized portfolio results based on the inputs. 

Screenshots of the dashboard can be found in the "Dashboard.png" file.

Project Setup

To get your project up and running, follow these steps to set up a virtual environment, install dependencies, and run the code.

1. Navigate to the Project Directory
First, open your terminal and navigate to the directory where your project is located. Replace /path/to/your/project with the actual path to your project:

cd /path/to/your/project

2. Create a Virtual Environment
Create a virtual environment to manage project-specific dependencies. This ensures that your project's packages don't interfere with system-wide packages or other projects:

python3 -m venv myenv


3. Activate the Virtual Environment
Activate the virtual environment. This step may vary depending on your operating system:

On macOS/Linux:
source myenv/bin/activate

On Windows:
myenv\Scripts\activate

After activation, your terminal prompt should change to indicate that the virtual environment is active.

4. Install Dependencies
With the virtual environment activated, install the necessary packages listed in requirements.txt. This file contains all the dependencies required for the project:

pip3 install -r requirements.txt

5. Run the Code
Now that all dependencies are installed, you can run the code. Replace your_script.py with the name of the Python file you want to execute:

python3 Static Portfolio Optimisation Dashboard.py

6. Deactivate the Virtual Environment
After you're done working, you can deactivate the virtual environment:

deactivate
