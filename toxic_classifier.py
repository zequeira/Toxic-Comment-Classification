import numpy as np
import pandas as pd
import texthero as hero
from texthero import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from utils import *

RANDOM_STATE = 42

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

    # Examples of comments corresponding to each category
    toxicity_labels = train_df.columns[2:]
    for i in toxicity_labels:
        print("{} :".format(i.upper()))
        print(train_df.loc[train_df[i] == 1, 'comment_text'].sample().values[0][:500], "\n")

    # Percentage of comments per label or category
    category_percentage(train_df)
    perc_clean_data = np.round(100 * train_df['clean'].sum() / train_df.shape[0], 2)
    print("{}% of the comments are clean i.e., non-toxic".format(perc_clean_data))

    # Correlation between labels
    corr_between_labels(train_df)

    # Comment length statistics across categories
    text_length_across_classes(train_df)

    # Word cloud
    generate_wordclouds(train_df, 'comment_text', 'all_toxic')
    generate_wordclouds(train_df, 'comment_text', 'clean')

    # Average word length in clean and toxic comments
    avg_word_len_plot(train_df)

    # Clean the comments
    train_df = clean_data(train_df, 'comment_text')
    # Remove empty comments rows
    train_df = train_df[train_df.comment_text != '']

    # Clean the comments keeping the stopwords
    # train_df['text_clean'] = hero.clean(train_df['comment_text'])
    # custom_pipeline = [preprocessing.fillna,
    #                    preprocessing.lowercase,
    #                    preprocessing.remove_digits,
    #                    preprocessing.remove_punctuation,
    #                    preprocessing.remove_diacritics,
    #                    preprocessing.remove_whitespace]
    # train_df['comment_text'] = hero.clean(train_df['comment_text'], custom_pipeline)

    # Re-check the average word length in clean and toxic comments after comment cleaning
    avg_word_len_plot(train_df)

    train_df['comment_text'] = train_df['comment_text'].apply(data_specific_preprocessing.long_word_fix)
    train_df['comment_text'] = train_df['comment_text'].apply(data_specific_preprocessing.repetitive_text_cleaning)

    # Re-check the average word length in clean and toxic comments after comment cleaning
    avg_word_len_plot(train_df)

    # Data Balance Analysis
    label_count(train_df)
    # The data is imbalanced, there are over 140k of non-toxic comments vs. a total of just 30k of toxic comments.
    # Therefore, lets sample a random set of 15300 clean comments.

    train_toxic = train_df[train_df[toxicity_labels].sum(axis=1) > 0]
    train_clean = train_df[train_df[toxicity_labels].sum(axis=1) == 0]

    train_df_balanced = pd.concat([
        train_toxic,
        train_clean.sample(n=15200, random_state=RANDOM_STATE)
    ])
    # Save data to +.csv for further use
    train_df_balanced.iloc[:, :-5].to_csv('./data/train_balanced.csv', sep=';')

    # Split the data into training and validation set for preliminary baseline analysis
    # train validation split
    tr_df, val_df = train_test_split(train_df_balanced, test_size=0.07, random_state=RANDOM_STATE)
    tr_df.shape, val_df.shape

    # Text representation using  using Term Frequency–Inverse Document Frequency (TF-IDF).
    # It is a numerical statistic intended to reflect how important a word is to a document in a collection or corpus.
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), analyzer='word',
                                       strip_accents='unicode', token_pattern=r'\w{1,}', use_idf=1,
                                       smooth_idf=1, sublinear_tf=1)

    train_tfidf = tfidf_vectorizer.fit_transform(tr_df['comment_text'])
    val_tfidf = tfidf_vectorizer.transform(val_df['comment_text'])

    classifiers = {}
    for target in toxicity_labels:
        clf = MultinomialNB()
        clf.fit(train_tfidf, tr_df[target])
        clf_val_pred = clf.predict(val_tfidf)
        print("Model for -{}- category".format(target))
        auc_score = roc_auc_score(val_df[target], clf_val_pred)
        print("validation auc: ", auc_score, "\n")
        classifiers['clf_'+target] = clf

    # Create new column with a string corresponding to the values of toxicity
    # train_df['toxicity'] = [''.join(map(str, toxic_int)) for toxic_int in train_df.iloc[:, 2:].values]

    # train_df['char_count_tcwsw'] = [len(c) for c in train_df['text_clean']]
    # train_df['char_count_tcwsw'].describe()
    # train_df['char_count_tcwsw'].min()
    # train_df['char_count_tcwsw'].max()
