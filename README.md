# 1D-CNN-for-chinese-text-NLP
Classification of Chinese medicine name entity

Run train.py you will get a trained model(using preprocessed data)

For test, you need some chinese medicine names and you need to cut these words using medicine vocabulary. Train a word2vec model use for create sentence vectors. Then transfer your test data's label into one-hot matrix.
Send your sentence-vectors matrix into model, you will get a predict matrix, compare with your true label matrix.