# Project Overview

This repository contains a set of Jupyter notebooks, data files, and resources designed to collect and process social media and financial data, perform exploratory analysis, and build machine learning models for topic and sentiment-based stock prediction.

## Contents

| Notebook/File                     | Description                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------- |
| `DataCollectionSocialMedia.ipynb` | Scripts and workflows to collect raw data from social media platforms (e.g., Reddit, Twitter).    |
| `ThirdMerge.ipynb`                | Data merging and preprocessing steps combining multiple sources into a unified dataset.           |
| `DataExploration&XGBoost.ipynb`   | Exploratory data analysis, feature engineering, and XGBoost model training for initial insights.  |
| `VaniallaModelTraining.ipynb`     | Baseline (vanilla) model training pipeline to benchmark performance before advanced methods.      |
| `LSTMmodel.ipynb`                 | Implementation and training of an LSTM-based model for sequence prediction or sentiment analysis. |
| `TopicModelling.ipynb`            | Topic modeling (e.g., LDA, NMF) to extract latent themes from text data.                          |
| `FineTuning.ipynb`                | Fine-tuning of pre-trained transformer models on the collected dataset for classification tasks.  |

## Data Files

| Data File                                | Description                                                                                                       |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `merged_stock_topic_data_3.csv`          | Combined social media topics with stock identifiers for initial merges.                                           |
| `cleaned_merged_stock_topic_data_3.csv`  | Cleaned and preprocessed version of `merged_stock_topic_data_3.csv`, ready for analysis.                          |
| `daily_stock_data_with_indices.csv`      | Historical daily stock prices with key market index values (e.g., S\&P 500, NASDAQ).                              |
| `daily_stock_data_with_indices_vol.csv`  | Daily stock prices augmented with trading volume and volatility metrics alongside market index values.            |
| `final_merged_stock_topic_sentiment.csv` | Final dataset merging social media topics, sentiment scores, stock data, indices, and volume/volatility features. |

## Prerequisites

* Python 3.8+
* Jupyter Notebook or JupyterLab

Install the required packages:

```bash
pip install -r requirements.txt
```

> **Note:** A sample `requirements.txt` can include:
>
> ```
> ```

pandas
numpy
scikit-learn
xgboost
torch
transformers
jupyter
gensim
nltk

```

Adjust versions as needed.

## Usage

1. **Data Collection**
   - Open `DataCollectionSocialMedia.ipynb` and follow the instructions to fetch raw social media data.
2. **Data Merging & Preprocessing**
   - Run `ThirdMerge.ipynb` to combine and clean the datasets, producing `merged_stock_topic_data_3.csv` and its cleaned variant.
3. **Exploratory Analysis & Baseline Training**
   - Use `DataExploration&XGBoost.ipynb` alongside the stock data files (`daily_stock_data_with_indices.csv`, `daily_stock_data_with_indices_vol.csv`) for feature engineering and XGBoost training.
   - Run `VaniallaModelTraining.ipynb` for baseline model comparisons.
4. **Modeling**
   - In `LSTMmodel.ipynb`, implement and train an LSTM network on `final_merged_stock_topic_sentiment.csv`.
   - Use `TopicModelling.ipynb` to extract latent themes from text data.
5. **Fine-Tuning**
   - Run `FineTuning.ipynb` to fine-tune transformer-based models on the final merged dataset.

## Project Structure

```

├── DataCollectionSocialMedia.ipynb
├── ThirdMerge.ipynb
├── DataExploration\&XGBoost.ipynb
├── VaniallaModelTraining.ipynb
├── LSTMmodel.ipynb
├── TopicModelling.ipynb
├── FineTuning.ipynb
├── merged\_stock\_topic\_data\_3.csv
├── cleaned\_merged\_stock\_topic\_data\_3.csv
├── daily\_stock\_data\_with\_indices.csv
├── daily\_stock\_data\_with\_indices\_vol.csv
├── final\_merged\_stock\_topic\_sentiment.csv
├── requirements.txt
└── README.md

```

## Next Steps

- Watch for additional data files and update the data files section accordingly.
- Ensure any new dependencies are added to `requirements.txt`.

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

Specify your project license here.

```
