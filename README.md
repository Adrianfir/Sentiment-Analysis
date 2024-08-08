# Sentiment Analysis Project

## Overview

This project aims to perform sentiment analysis using a combination of text preprocessing techniques, tokenization, padding, and an LSTM model for binary sentiment classification. The project is structured to build a compehensive machine learning pipeline using Scikit-learn and TensorFlow, ensuring a modular and easily extendable codebase.

## Features

- **Text Preprocessing**: Cleaning text data, removing stop words, stemming, and lemmatization.
- **Tokenization**: Converting text into sequences of tokens.
- **Padding**: Ensuring uniform input length by padding sequences.
- **Model Building**: Creating and training an LSTM model for sentiment classification.
- **Pipeline Integration**: Combining all steps into a single Scikit-learn pipeline for ease of use.
- **Model Evaluation**: Evaluating the model using accuracy metric on the test dataset.

## Usage

To run this project, download Sentiment140.csv dataset and change the path in parameters.json file. Then use "python main.py" command in the terminal.

This script will:

1-Load the dataset.
2-Split the data into training and testing sets.
3-Preprocess the text.
4-Tokenize and pad the sequences.
5-Build and train an LSTM model.
6-Evaluate the model on the test set.
7-Save the trained pipeline to a file.

## Project Structure

- main.py: Main script to run the sentiment analysis pipeline.
- build_model.py: Defines the LSTM model using TensorFlow.
- padding.py: Handles padding of tokenized sequences.
- prep_text.py: Preprocesses the text data.
- stop_stem_lemmat.py: Handles stop words removal, stemming, and lemmatization.
- tokenizing.py: Handles tokenization of text data.
- config/: Contains configuration files for the project.
- util/: Contains utility functions used in the project.
