# Stock Prediction

These are just some work in progress deep learning/machine learning stock prediction projects I have been working on. I have built over 20 different models but these are the two that perform the best 

# DRSE XGBoost
I got the idea for the DRSE from the research paper found at: https://www.sciencedirect.com/science/article/pii/S0925231218303540. It depicts using random subspaces of the input text data to train multiple XGBoost models, then combine the predictions to reduce the randomness/noise of the models. I changed it as I instead get random subsamples from the technical indicators rather then textual data. I also changed the model to be regression instead of classification.

# Lasso LSTM
I got the idea for the Lasso LSTM from the research paper found at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9680880/. It depicts using a LASSO reduction on technical indicators and textual data to extract the most important ones and then predicting with LSTM on those. My model differs as I don't use textual data or technical indicators, instead I use LASSO reduction on each input 'step' for the LSTM there for extracting only the close prices that relate most to the target. Therefore reducing the noise of the data.  
