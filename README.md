# Traffic Flow Prediction Project

## Overview
This project aims to predict traffic flow patterns using machine learning models. The goal is to analyze traffic volume at different junctions and times to help in developing smarter traffic management systems. We utilize two primary predictive models: ARIMA (AutoRegressive Integrated Moving Average) and LSTM (Long Short-Term Memory networks), to evaluate their performance in predicting urban traffic flows.

## Files Description

### `main.ipynb`
This Jupyter notebook contains the main analysis and modeling work for the project. It includes data loading, preprocessing, model training, and evaluation. The notebook is structured to walk through the process step-by-step, from initial data exploration to final model performance comparison.

### `modules.py`
This Python script file contains custom functions and classes used within the main notebook. Functions in this module are used for data preprocessing, outlier detection, model parameter selection, and plotting results. Importing this module in the notebook helps keep the code clean and more organized.

### `traffic.csv`
The dataset used in this project, sourced from Kaggle, comprises hourly traffic counts at multiple junctions over a span of two years (2015-2017). Attributes include `DateTime`, `Junction`, `Vehicles`, and a unique `ID` for each record. This file serves as the primary input for our analysis.

## Setup and Installation
To run this project locally, you'll need to have Python installed, along with several libraries including pandas, numpy, matplotlib, seaborn, sklearn, and statsmodels. You can install these packages via pip:

```pip install pandas numpy matplotlib seaborn sklearn statsmodels```


For handling LSTM models, ensure that you have TensorFlow installed:

```pip install tensorflow```


## Usage
1. Clone this repository or download the files into your local machine.
2. Open the `main.ipynb` notebook in a Jupyter environment.
3. Ensure that `modules.py` and `traffic.csv` are in the same directory as the notebook for proper functionality.
4. Run the cells in `main.ipynb` sequentially to perform the analysis and view the results.

## Contributing
Contributions to this project are welcome. You can contribute in several ways:
- Improving the efficiency of the code
- Expanding the analysis with more models or features
- Enhancing visualization
To contribute, please fork the repository and submit a pull request.

## License
This project is open source and available under the [MIT License](LICENSE.md).




