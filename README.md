# Spam Detection Project

This project aims to detect spam messages using various machine learning techniques. It includes data preprocessing, model training, evaluation, and real-time Gmail message analysis with a word cloud visualization.

## Features

- **Data Preprocessing**: Cleaning and preparing the dataset for training.
- **Model Training**: Training multiple machine learning models to classify spam and non-spam messages.
- **Model Evaluation**: Evaluating the performance of the trained models using metrics like accuracy, precision, recall, and F1 score.
- **Real-time Gmail Analysis**: Analyzing Gmail messages in real-time to detect spam and visualize the results using a word cloud.
- **Robot Integration**: Automated bot for fetching and processing Gmail messages.

## Installation

To run the project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/RAJPUTRoCkStAr/Spam-detection.git
    cd spam-detection
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Code**:
    ```bash
    streamlit run app.py
    ```



## Usage

The project provides several scripts for different stages of the spam detection process:

1. **Data Preprocessing**
2. **Model Training**
3. **Model Evaluation**
4. **Prediction**
5. **Real-time Gmail Analysis**
6. **Word Cloud Visualization**
7. **Jarvis**
For live usage and demo, visit [Spam Detection Project Demo](https://spam-detection-ml.streamlit.app/).


## Data Files

Ensure that the following data files are available in the `data/` directory:

- `spam.csv`: The dataset containing spam and non-spam messages.

## Custom Functions

The project uses custom functions to preprocess data, train models, and make predictions:

- `preprocess_data()`: Cleans and preprocesses the dataset.
- `train_model()`: Trains the machine learning models.
- `evaluate_model()`: Evaluates the performance of the trained models.
- `predict()`: Predicts whether a given message is spam or not.
- `fetch_gmail_messages()`: Fetches messages from Gmail using the Gmail API.
- `generate_wordcloud()`: Generates a word cloud from the fetched Gmail messages.

## License

This project is licensed under the MIT License.

