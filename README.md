# Amazon Reviews Sentiment Analysis

This repo contains a notebook for a Sentiment Analysis Model on the Amazon Videogames Reviews dataset.
The main parts of the notebook are:
1. Data Reading and Cleaning
2. Small EDA
3. Logistic Regression model with TF-IDF vectorization
4. PyTorch Model building with HuggingFace APIs (DistilBert Base Uncased model)
5. Evaluation of the test predictions

## Data and EDA

During this phase a small portion of data, corresponding to the 10% of the dataset, was used in order to train and validate the model. In this way only 360.000 text samples were considered.
These text were however raw data. In the notebook there is then a preprocessing step that parses the string, extracts the label (0 or 1 for negative and positive, respectively),
then a cleaning of the text is performed (Cleaning special characters, removing stop word, also stemming or lemmization can be performed).

After that, an analysis of the (tokenized) sequence length distribution for the descriptions in the dataset is carried on, obtaninig the following plots:

<img src=""></img></br>
<img src=""></img></br>

## Logistic Regression model with TF-IDF

This simple LR model takes as imput some features obtained from the Scikit-Learn callable TfidfVectorizer, that given a text creates: a TF-IDF map for tokens, but also bi and trigrams, and given the presence of a token in the string flags as 1 the corresponding token. Of course this doesn't give any attention to the order of the tokens, so it is a very simple yet powerful model. In fact, it reaches an accuracy of 86% on both training and validation data.

## Using DistilBert HuggingFace APIs

After that in order to perform better, I used an Autoconfigured and Pretrained DistilBert model. I built it using PyTorch.

