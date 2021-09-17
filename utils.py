import re
import string
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from wordsegment import load, segment
load()


plt.style.use("seaborn-pastel")

def category_percentage(df):
    df['clean'] = np.where(
        (df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (
                    df['insult'] == 0) & (df['identity_hate'] == 0), 1, 0)

    categories = ['toxic', 'severe_toxic', 'obscene', 'threat',
                  'insult', 'identity_hate', 'clean']
    plot_data = df[categories].mean() * 100

    plt.figure(figsize=(10, 5))
    plt.title("percentage records by category")
    sns.barplot(x=plot_data.index, y=plot_data.values)
    plt.show()
    return


def label_count(df):
    label_columns = df.columns.tolist()[2:8]
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat',
                  'insult', 'identity_hate', 'clean']
    plot_data = df[categories].sum()

    plt.figure(figsize=(10, 5))
    plt.title("Number of comments per category")
    sns.barplot(x=plot_data.index, y=plot_data.values)
    plt.show()

    plt.figure(figsize=(10, 5))
    df[label_columns].sum().sort_values().plot(kind='barh')
    print(df[label_columns].sum().sort_values())
    plt.show()
    return


def text_length_across_classes(df):
    df['comment_length'] = df['comment_text'].apply(lambda x: len(x.split()))

    median_text_len = []
    mean_text_len = []
    min_text_len = []
    max_text_len = []
    max_distinct_tokens = []

    for i in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        mean_text_len.append(df[df[i] == 1]['comment_length'].mean())
        min_text_len.append(df[df[i] == 1]['comment_length'].min())
        max_text_len.append(df[df[i] == 1]['comment_length'].max())
        median_text_len.append(df[df[i] == 1]['comment_length'].median())
        df['distinct_tokens'] = df['comment_text'].apply(lambda x: len(set(x.split())))
        max_distinct_tokens.append(df[df[i] == 1]['distinct_tokens'].max())

    mean_text_len.append(df[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (
                df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]['comment_length'].mean())
    min_text_len.append(df[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (
                df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]['comment_length'].min())
    max_text_len.append(df[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (
                df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]['comment_length'].max())
    median_text_len.append(df[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (
                df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]['comment_length'].median())
    max_distinct_tokens.append(df[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (
                df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]['distinct_tokens'].max())

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    sns.barplot(ax=axes[0, 0], x=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'clean'],
                y=median_text_len)
    axes[0, 0].set_title('median text length')
    sns.barplot(ax=axes[0, 1], x=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'clean'],
                y=min_text_len)
    axes[0, 1].set_title('minimum text length')
    sns.barplot(ax=axes[1, 0], x=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'clean'],
                y=max_text_len)
    axes[1, 0].set_title('max text length')
    sns.barplot(ax=axes[1, 1], x=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'clean'],
                y=max_distinct_tokens)
    axes[1, 1].set_title('max distinct tokens')

    fig.suptitle('text length statistics')
    plt.show()
    return


def corr_between_labels(df):
    plt.figure(figsize=(15, 8))
    plt.title("correlation between toxic categories")
    sns.heatmap(df.corr(), cmap='YlGnBu', annot=True)
    plt.show()
    return


## Gram statistics
def gram_analysis(data, gram):
    stop_words_set = set(stopwords.words('english'))
    tokens = [t for t in data.lower().split(" ") if t != "" if t not in stop_words_set]
    ngrams = zip(*[tokens[i:] for i in range(gram)])
    final_tokens = [" ".join(z) for z in ngrams]
    return final_tokens


def gram_freq(df, gram, categ_col, text_col):
    category_text = " ".join(df[df[categ_col] == 1][text_col].sample(200).values)
    toks = gram_analysis(category_text, gram)
    tok_freq = pd.DataFrame(data=[toks, np.ones(len(toks))]).T.groupby(0).sum().reset_index()
    tok_freq.columns = ['token', 'frequency']
    tok_freq = tok_freq.sort_values(by='frequency', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.title("{} most common tokens".format(categ_col))
    sns.barplot(x='token', y='frequency', data=tok_freq.iloc[:30])
    plt.xticks(rotation=90)
    plt.show()

    return


def avg_word_len_plot(df):
    # word distribution across categories
    df['punct_count'] = df['comment_text'].apply(lambda x: len([a for a in x if a in string.punctuation]))
    df['avg_word_length'] = df['comment_text'].apply(lambda x: np.round(np.mean([len(a) for a in x.split()])))

    clean = df[df['clean'] == 1].avg_word_length.value_counts().reset_index()
    clean.columns = ['length', 'frequency']
    print("clean comments max token length : {}".format(max(clean.length)))
    clean = clean.sort_values(by='length')
    plt.figure(figsize=(20, 7))
    plt.title("Average word length - clean comments")
    sns.barplot(x=clean.length, y=clean.frequency)
    plt.xticks(rotation=90)
    plt.show()

    toxic = df[df['clean'] == 0].avg_word_length.value_counts().reset_index()
    toxic.columns = ['length', 'frequency']
    print("toxic comments max token length : {}".format(max(toxic.length)))
    toxic = toxic.sort_values(by='length')
    plt.figure(figsize=(20, 7))
    plt.title("Average word length -toxic comments (all forms)")
    sns.barplot(x=toxic.length, y=toxic.frequency)
    plt.xticks(rotation=90)
    plt.show()
    return


def generate_wordclouds(df, text_col, categ_col):
    df['clean'] = np.where(
        (df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (
                    df['insult'] == 0) & (df['identity_hate'] == 0), 1, 0)

    if categ_col == 'all_toxic':
        category_text = df[df['clean'] != 1][text_col].values
    else:
        category_text = df[df[categ_col] == 1][text_col].values

    plt.figure(figsize=(15, 8))
    wc = WordCloud(background_color="black",
                   max_words=5000,
                   stopwords=STOPWORDS,
                   collocations=False,
                   max_font_size=40)
    wc.generate(" ".join(category_text))
    plt.title("{} word cloud".format(categ_col), fontsize=20)
    # plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)
    plt.imshow(wc.recolor(colormap='Pastel2', random_state=17), alpha=0.98)
    plt.axis('off')
    plt.show()
    return


# def venn_(df):
#     figure, axes = plt.subplots(2, 2, figsize=(20, 20))
#     toxic = set(df[df['toxic'] == 1].index)
#     severe_toxic = set(df[df['severe_toxic'] == 1].index)
#     obscene = set(df[df['obscene'] == 1].index)
#     threat = set(df[df['threat'] == 1].index)
#     insult = set(df[df['insult'] == 1].index)
#     identity_hate = set(df[df['identity_hate'] == 1].index)
#     clean = set(df[df['clean'] == 1].index)
#
#     v1 = venn3([toxic, severe_toxic, obscene],
#                set_labels=('Toxic', 'Severe toxic', 'Obscene'), set_colors=('#a5e6ff', '#3c8492', '#9D8189'),
#                ax=axes[0][0])
#     for text in v1.set_labels:
#         text.set_fontsize(22)
#     v2 = venn3([threat, insult, identity_hate],
#                set_labels=('Threat', 'Insult', 'Identity hate'), set_colors=('#e196ce', '#F29CB7', '#3c81a9'),
#                ax=axes[0][1])
#     for text in v2.set_labels:
#         text.set_fontsize(22)
#     v3 = venn3([toxic, insult, obscene],
#                set_labels=('Toxic', 'Insult', 'Obscene'), set_colors=('#a5e6ff', '#F29CB7', '#9D8189'), ax=axes[1][0])
#     for text in v3.set_labels:
#         text.set_fontsize(22)
#     v4 = venn3([threat, identity_hate, obscene],
#                set_labels=('Threat', 'Identity hate', 'Obscene'), set_colors=('#e196ce', '#3c81a9', '#9D8189'),
#                ax=axes[1][1])
#     for text in v4.set_labels:
#         text.set_fontsize(22)
#     plt.show()
#
#     # deleting used variables
#     del toxic
#     del severe_toxic
#     del obscene
#     del threat
#     del insult
#     del identity_hate
#     del clean
#     return


def meta_data_analysis(df, text_col):
    meta_df = pd.DataFrame()
    meta_df['punctuations'] = df[text_col].apply(lambda x: len([a for a in str(x) if a in string.punctuation]))
    meta_df['hashtags'] = df[text_col].apply(lambda x: len([a for a in x.split() if a.startswith("#")]))
    meta_df['usernames'] = df[text_col].apply(lambda x: len([a for a in x.split() if a.startswith("@")]))
    meta_df['stop_words'] = df[text_col].apply(lambda x: len([a for a in x.lower().split() if a in STOPWORDS]))
    meta_df['upper_case_words'] = df[text_col].apply(lambda x: len([a for a in x.split() if a.isupper()]))
    meta_df['urls'] = df[text_col].apply(lambda x: len([a for a in x.split() if a.startswith(tuple(['http', 'www']))]))
    meta_df['word_count'] = df[text_col].apply(lambda x: len(x.split()))
    meta_df['distinct_word_count'] = df[text_col].apply(lambda x: len(set(x.split())))
    meta_df['clean'] = df['clean'].copy()
    return meta_df


## Text cleaning
class TextCleaningUtils:
    '''
        This class contains implementations of various text cleaning operations (Static Methods)
    '''

    @staticmethod
    def expand_abbreviations(text):
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"there's", "there is", text)
        text = re.sub(r"We're", "We are", text)
        text = re.sub(r"That's", "That is", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"they're", "they are", text)
        text = re.sub(r"Can't", "Cannot", text)
        text = re.sub(r"wasn't", "was not", text)
        text = re.sub(r"don\x89Ûªt", "do not", text)
        text = re.sub(r"aren't", "are not", text)
        text = re.sub(r"isn't", "is not", text)
        text = re.sub(r"What's", "What is", text)
        text = re.sub(r"haven't", "have not", text)
        text = re.sub(r"hasn't", "has not", text)
        text = re.sub(r"There's", "There is", text)
        text = re.sub(r"He's", "He is", text)
        text = re.sub(r"It's", "It is", text)
        text = re.sub(r"You're", "You are", text)
        text = re.sub(r"I'M", "I am", text)
        text = re.sub(r"shouldn't", "should not", text)
        text = re.sub(r"wouldn't", "would not", text)
        text = re.sub(r"couldn't", "could not", text)
        text = re.sub(r"i'm", "I am", text)
        text = re.sub(r"I\x89Ûªm", "I am", text)
        text = re.sub(r"I'm", "I am", text)
        text = re.sub(r"Isn't", "is not", text)
        text = re.sub(r"Here's", "Here is", text)
        text = re.sub(r"you've", "you have", text)
        text = re.sub(r"you\x89Ûªve", "you have", text)
        text = re.sub(r"we're", "we are", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"couldn't", "could not", text)
        text = re.sub(r"we've", "we have", text)
        text = re.sub(r"it\x89Ûªs", "it is", text)
        text = re.sub(r"doesn\x89Ûªt", "does not", text)
        text = re.sub(r"It\x89Ûªs", "It is", text)
        text = re.sub(r"Here\x89Ûªs", "Here is", text)
        text = re.sub(r"who's", "who is", text)
        text = re.sub(r"I\x89Ûªve", "I have", text)
        text = re.sub(r"y'all", "you all", text)
        text = re.sub(r"can\x89Ûªt", "cannot", text)
        text = re.sub(r"would've", "would have", text)
        text = re.sub(r"it'll", "it will", text)
        text = re.sub(r"we'll", "we will", text)
        text = re.sub(r"wouldn\x89Ûªt", "would not", text)
        text = re.sub(r"We've", "We have", text)
        text = re.sub(r"he'll", "he will", text)
        text = re.sub(r"Y'all", "You all", text)
        text = re.sub(r"Weren't", "Were not", text)
        text = re.sub(r"Didn't", "Did not", text)
        text = re.sub(r"they'll", "they will", text)
        text = re.sub(r"DON'T", "DO NOT", text)
        text = re.sub(r"That\x89Ûªs", "That is", text)
        text = re.sub(r"they've", "they have", text)
        text = re.sub(r"they'd", "they would", text)
        text = re.sub(r"i'd", "I would", text)
        text = re.sub(r"should've", "should have", text)
        text = re.sub(r"You\x89Ûªre", "You are", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"Don\x89Ûªt", "Do not", text)
        text = re.sub(r"i'll", "I will", text)
        text = re.sub(r"weren't", "were not", text)
        text = re.sub(r"They're", "They are", text)
        text = re.sub(r"Can\x89Ûªt", "Cannot", text)
        text = re.sub(r"you\x89Ûªll", "you will", text)
        text = re.sub(r"I\x89Ûªd", "I would", text)
        text = re.sub(r"let's", "let us", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"don't", "do not", text)
        text = re.sub(r"you're", "you are", text)
        text = re.sub(r"i've", "I have", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"i'll", "I will", text)
        text = re.sub(r"doesn't", "does not", text)
        text = re.sub(r"i'd", "I would", text)
        text = re.sub(r"didn't", "did not", text)
        text = re.sub(r"ain't", "am not", text)
        text = re.sub(r"you'll", "you will", text)
        text = re.sub(r"I've", "I have", text)
        text = re.sub(r"Don't", "do not", text)
        text = re.sub(r"I'll", "I will", text)
        text = re.sub(r"I'LL", "I will", text)
        text = re.sub(r"I'd", "I would", text)
        text = re.sub(r"Let's", "Let us", text)
        text = re.sub(r"you'd", "You would", text)
        text = re.sub(r"It's", "It is", text)
        text = re.sub(r"Ain't", "am not", text)
        text = re.sub(r"Haven't", "Have not", text)
        text = re.sub(r"Hadn't", "Had not", text)
        text = re.sub(r"Could've", "Could have", text)
        text = re.sub(r"youve", "you have", text)
        text = re.sub(r"donå«t", "do not", text)

        return text

    cleaning_regex_map = {
        'web_links': r'(?i)(?:(?:http(?:s)?:)|(?:www\.))\S+',
        'email': r'[\w.]+@\w+\.[a-z]{3}',
        'twitter_handles': r'[#@]\S+',
        'redundant_newlines': r'[\r|\n|\r\n]+',
        'redundant_spaces': r'\s\s+',
        'punctuations': r'[\.,!?;:]+',
        #         'special_chars': r'[^a-zA-Z0-9\s\.,!?;:]+',
        'special_chars': r'[^a-zA-Z\s\.,!?;:]+'  ## removing nums

    }

    @staticmethod
    def clean_text_from_regex(text, text_clean_regex):
        '''
            Follow a particular cleaning expression, provided
            as an input by an user to clean the text.
        '''

        text = text_clean_regex.sub(' ', text).strip()
        return text

    @staticmethod
    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    @staticmethod
    def remove_web_links(text):
        '''
            Removes any web link that follows a particular default expression,
            present in the text.
        '''

        web_links_regex = re.compile(TextCleaningUtils.cleaning_regex_map['web_links'])
        text = TextCleaningUtils.clean_text_from_regex(text, web_links_regex)
        return text

    @staticmethod
    def remove_email_addresses(text):
        '''
            Removes email addresses present in the text.
        '''

        email_regex = re.compile(TextCleaningUtils.cleaning_regex_map['email'])
        text = TextCleaningUtils.clean_text_from_regex(text, email_regex)
        return text

    @staticmethod
    def remove_twitter_handles(text):
        '''
            Removes any twitter handle present in the text.
        '''

        twitter_handles_regex = re.compile(TextCleaningUtils.cleaning_regex_map['twitter_handles'])
        text = TextCleaningUtils.clean_text_from_regex(text, twitter_handles_regex)
        return text

    @staticmethod
    def remove_emojis(text):
        emoji_clean = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 u"\U00002702-\U000027B0"
                                 u"\U000024C2-\U0001F251"
                                 "]+", flags=re.UNICODE)
        text = emoji_clean.sub(r'', text)
        url_clean = re.compile(r"https://\S+|www\.\S+")
        text = url_clean.sub(r'', text)
        return text

    @staticmethod
    def remove_redundant_newlines(text):
        '''
            Removes any redundant new line present in the text.
        '''

        redundant_newlines_regex = re.compile(
            TextCleaningUtils.cleaning_regex_map['redundant_newlines'])
        text = TextCleaningUtils.clean_text_from_regex(text, redundant_newlines_regex)
        return text

    @staticmethod
    def remove_redundant_spaces(text):
        '''
            Remove any redundant space provided as default,
            that is present in the text.
        '''

        redundant_spaces_regex = re.compile(
            TextCleaningUtils.cleaning_regex_map['redundant_spaces'])
        text = TextCleaningUtils.clean_text_from_regex(text, redundant_spaces_regex)
        return text

    @staticmethod
    def remove_punctuations(text):
        '''
            Removes any punctuation that follows the default expression, in the text.
        '''

        remove_punctuations_regex = re.compile(TextCleaningUtils.cleaning_regex_map['punctuations'])
        text = TextCleaningUtils.clean_text_from_regex(text, remove_punctuations_regex)
        return text

    @staticmethod
    def remove_special_chars(text):
        '''
            Replace any special character provided as default,
            which is present in the text with space
        '''

        special_chars_regex = re.compile(TextCleaningUtils.cleaning_regex_map['special_chars'])
        text = TextCleaningUtils.clean_text_from_regex(text, special_chars_regex)
        return text

    @staticmethod
    def remove_exaggerated_words(text):
        '''
            Removes any exaggerated word present in the text.
        '''

        return ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))

    @staticmethod
    def replace_multiple_chars(text):
        '''
            Replaces multiple characters present in the text.
        '''

        char_list = ['.', '?', '!', '#', '$', '/', '@', '*', '(', ')', '+']
        final_text = ''
        for i in char_list:
            if i in text:
                pattern = "\\" + i + '{2,}'
                repl_str = i.replace("\\", "")
                text = re.sub(pattern, repl_str, text)
                final_text = ' '.join(text.split())
        return final_text

    @staticmethod
    def replace_sign(text):
        '''
            Replaces any sign with words like & with 'and', in the text.
        '''
        sign_list = {'&': ' and ', '/': ' or ', '\xa0': ' '}
        final_text = ''
        for i in sign_list:
            if i in text:
                text = re.sub(i, sign_list[i], text)
                final_text = ' '.join(text.split())
        return final_text

    @staticmethod
    def remove_accented_char(text):
        text = unicodedata.normalize('NFD', text) \
            .encode('ascii', 'ignore') \
            .decode("utf-8")
        return str(text)

    @staticmethod
    def replace_characters(text, replace_map):
        '''
            Replaces any character custom provided by an user.
        '''

        for char, replace_val in replace_map.items():
            text = text.replace(char, replace_val)
        return text


def clean_data(df, col_to_clean):
    df[col_to_clean] = df[col_to_clean].apply(TextCleaningUtils.remove_web_links)
    df[col_to_clean] = df[col_to_clean].apply(TextCleaningUtils.remove_email_addresses)
    df[col_to_clean] = df[col_to_clean].apply(TextCleaningUtils.remove_twitter_handles)
    df[col_to_clean] = df[col_to_clean].apply(TextCleaningUtils.expand_abbreviations)
    df[col_to_clean] = df[col_to_clean].apply(TextCleaningUtils.remove_emojis)
    df[col_to_clean] = df[col_to_clean].apply(TextCleaningUtils.remove_special_chars)
    df[col_to_clean] = df[col_to_clean].apply(TextCleaningUtils.remove_redundant_spaces)
    df[col_to_clean] = df[col_to_clean].apply(TextCleaningUtils.remove_punctuations)
    df[col_to_clean] = df[col_to_clean].apply(TextCleaningUtils.remove_exaggerated_words)
    df[col_to_clean] = df[col_to_clean].apply(TextCleaningUtils.remove_redundant_newlines)
    df[col_to_clean] = df[col_to_clean].astype(str)
    df[col_to_clean] = df[col_to_clean].str.lower()
    return df


class data_specific_preprocessing:
    @staticmethod
    def long_word_cleaning(wrd):
        if wrd.startswith('haha') | wrd.startswith('ahah'):
            wrd = 'haha'
        elif wrd.startswith('lol') | wrd.startswith('olo'):
            wrd = 'lol'
        elif wrd.startswith('fuckfuck') | wrd.startswith('uckfuc') | wrd.startswith('ckfuck') | wrd.startswith(
                'kfuckf'):
            wrd = 'fuck'
        elif wrd.startswith('suck') | wrd.startswith('ucks') | wrd.startswith('cksu') | wrd.startswith('ksuc'):
            wrd = 'suck'
        elif wrd.startswith('mwahaha') | wrd.startswith('muahaha'):
            wrd = 'muahahaha'
        elif wrd.startswith('bwahaha'):
            wrd = 'bwahaha'
        elif wrd.startswith('cunt') | wrd.startswith('untc') | wrd.startswith('ntcu') | wrd.startswith('tcun'):
            wrd = 'cunt'
        elif wrd.startswith('blah'):
            wrd = 'blah'
        elif wrd.startswith('tytyty'):
            wrd = 'ty'
        return wrd

    @staticmethod
    def long_word_fix(text):
        x = text.split()
        # fixes long words
        x = [data_specific_preprocessing.long_word_cleaning(wrd) if len(wrd) > 20 else wrd for wrd in x]
        # returns words from spaceless phrases
        text = " ".join([" ".join(segment(wrd)) if len(wrd) > 20 else wrd for wrd in x])
        return text

    @staticmethod
    def repetitive_text_cleaning(text):
        x = text.split()
        if len(x) > 100 and len(set(x)) <= 30:
            text = " ".join(x[:len(set(x)) + 10])
        return text