import numpy as np
import pandas as pd
import texthero as hero
from texthero import preprocessing
import matplotlib.pyplot as plt


def load_data(file_name, separator=';'):
    try:
        dataDF = pd.read_csv(file_name, sep=separator)
        print('FILE EXIST')
        return dataDF
    except IOError as ioe:
        # file didn't exist (or other issues)
        print('File do not exist!')
        print(ioe)
        return False


if __name__ == '__main__':
    train_df = load_data('data/train.csv', ',')
    test_df = load_data('data/test.csv', ',')
    test_labels_df = load_data('data/test_labels.csv', ',')

    # Create new column with a string corresponding to the values of toxicity
    train_df['toxicity'] = [''.join(map(str, toxic_int)) for toxic_int in train_df.iloc[:, 2:].values]

    train_df['text_clean'] = hero.clean(train_df['comment_text'])

    custom_pipeline = [preprocessing.fillna,
                       preprocessing.lowercase,
                       preprocessing.remove_digits,
                       preprocessing.remove_punctuation,
                       preprocessing.remove_diacritics,
                       preprocessing.remove_whitespace]

    train_df['text_clean_with_stopW'] = hero.clean(train_df['comment_text'], custom_pipeline)

    # Text representation using  using Term Frequency–Inverse Document Frequency (TF-IDF).
    # It is a numerical statistic intended to reflect how important a word is to a document in a collection or corpus.
    # train_df['text_tfidf'] = (hero.tfidf(train_df['text_clean']))
    train_df['text_tfidf'] = (hero.tfidf(train_df['text_clean'], max_features=2500))

    train_df['char_count_tcwsw'] = [len(c) for c in train_df['text_clean_with_stopW']]
    train_df['char_count_tcwsw'].describe()
    train_df['char_count_tcwsw'].min()
    train_df['char_count_tcwsw'].max()

    train_df = train_df[train_df.char_count_tcwsw != 0]

    train_df.iloc['pca'] = hero.pca(train_df['text_tfidf'])

    hero.scatterplot(
        train_df,
        col='pca',
        color='toxicity',
        title="Toxicity Wikipedia in Comments"
    )
    plt.show()

    import torch

    x = torch.rand(5, 3)
    print(x)

    torch.cuda.is_available()

