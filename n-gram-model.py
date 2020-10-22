import os
import string
import nltk
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends
from bs4 import BeautifulSoup
import math
from collections import Counter
import matplotlib.pyplot

lemmatizer = nltk.WordNetLemmatizer()


def clean(text):
    """Make all text lower and return it."""
    cleaned = str(text).lower()
    return cleaned


def sent_tokenize(sentence, n):
    """Return tokens of words from a sentence, n indicates n-gram, for padding purposes."""
    cleaned = clean(sentence)
    sent = cleaned.translate(str.maketrans('', '', string.punctuation))
    sent_tokens = nltk.word_tokenize(sent)
    tokens = []
    if n == 1:
        sent_tokens = pad_both_ends(sent_tokens, 1)
    if n == 2:
        sent_tokens = pad_both_ends(sent_tokens, 2)
    if n == 3:
        sent_tokens = pad_both_ends(sent_tokens, 3)
    if n == 4:
        sent_tokens = pad_both_ends(sent_tokens, 4)
    sent_tokens = [lemmatizer.lemmatize(token) for token in sent_tokens]
    tokens.extend(sent_tokens)
    return tokens


def tokenize(text, n):
    """Return tokens of words, n indicates n-gram, for padding purposes."""
    sentences = nltk.sent_tokenize(text)
    tokens = []
    for sent in sentences:
        sent = sent.translate(str.maketrans('', '', string.punctuation))
        sent_tokens = nltk.word_tokenize(sent)
        if n == 1:
            sent_tokens = pad_both_ends(sent_tokens, 1)
        if n == 2:
            sent_tokens = pad_both_ends(sent_tokens, 2)
        if n == 3:
            sent_tokens = pad_both_ends(sent_tokens, 3)
        if n == 4:
            sent_tokens = pad_both_ends(sent_tokens, 4)
        sent_tokens = [lemmatizer.lemmatize(token) for token in sent_tokens]
        tokens.extend(sent_tokens)
    return tokens


# Empty lists to store training and test grams
train_1grams = []
train_2grams = []
train_3grams = []
train_4grams = []
test_1grams = []
test_2grams = []
test_3grams = []
test_4grams = []


def train(directory, x):
    """Iterate through the main directory. Obtain a list of sub-folders.
    Parse files, collect and clean tokens from files.
    Writes to the lists above.
    x should be an integer (0 or 1 indicating if test or
    training, 0 for training and 1 for test."""
    folders = []  # Store the sub-folders
    main_folder = os.listdir(directory)
    for sub in main_folder:
        if sub != ".DS_Store":
            folders.append(directory + sub)
    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            with open(folder + '/' + file, "rb") as f:
                contents = f.read()
                soup = BeautifulSoup(contents, 'html.parser')
                text = soup.get_text()
                cleaned = clean(text)
                if x == 0:
                    train_1grams.extend(list(ngrams(tokenize(cleaned, 1), 1)))
                    train_2grams.extend(list(ngrams(tokenize(cleaned, 2), 2)))
                    train_3grams.extend(list(ngrams(tokenize(cleaned, 3), 3)))
                    train_4grams.extend(list(ngrams(tokenize(cleaned, 4), 4)))
                if x == 1:
                    test_1grams.extend(list(ngrams(tokenize(cleaned, 1), 1)))
                    test_2grams.extend(list(ngrams(tokenize(cleaned, 2), 2)))
                    test_3grams.extend(list(ngrams(tokenize(cleaned, 3), 3)))
                    test_4grams.extend(list(ngrams(tokenize(cleaned, 4), 4)))


# Run the training function on our data
train('Dataset/TrainingSet/', 0)
train('Dataset/TestSet/', 1)

# Make counters of the data for easier calculations
count_train_1grams = Counter(train_1grams)
count_train_2grams = Counter(train_2grams)
count_train_3grams = Counter(train_3grams)
count_train_4grams = Counter(train_4grams)
count_test_1grams = Counter(test_1grams)
count_test_2grams = Counter(test_2grams)
count_test_3grams = Counter(test_3grams)
count_test_4grams = Counter(test_4grams)

# Some frequency distributions for analysis
test_freq_dist_1gram = nltk.FreqDist(test_1grams)
train_freq_dist_1gram = nltk.FreqDist(train_1grams)


def count_files(directory):
    """A simple function that opens a main directory and counts the number of files located in the directory's
    sub-folders. Returns an integer."""
    folders = []
    file_count = 0
    main_folder = os.listdir(directory)  # Opens main directory
    for sub in main_folder:  # Iterates through directory to create a list of sub-directory paths
        if sub != ".DS_Store":
            folders.append(directory + sub)
    for folder in folders:  # Opens all sub-directories
        files = os.listdir(folder)
        file_count += len(files)
    return file_count


def count_sentences(file):
    """Simple function that counts the amount of sentences in all files contained in the directory.
    Takes a directory and iterates through sub-folders to open files contained in them.
    Returns integer."""
    sentence_count = 0
    folders = []
    main_folder = os.listdir(file)  # Opens main directory
    for sub in main_folder:  # Iterates through directory to create a list of sub-directory paths
        if sub != ".DS_Store":
            folders.append(file + sub)
    for folder in folders:  # Opens all sub-directories
        files = os.listdir(folder)
        for file in files:
            with open(folder + '/' + file, "rb") as f:
                contents = f.read().decode("utf-8")
                soup = BeautifulSoup(contents, 'html.parser')
                text = soup.get_text()
                cleaned = clean(text)
                sentences = nltk.sent_tokenize(cleaned)
                sentence_count += len(sentences)
    return sentence_count


# bigram (w1, w2)
# tc = count in training data
def bigram_probability(test_bigram, train_bigram_counter, train_1gram_counter, smoothing=True):
    """Returns a float calculating the probability for an individual 3-gram.
    Default settings smooths data, set smoothing=False for unsmoothed probability."""
    w1_count = train_1gram_counter[test_bigram[0:1]]
    gram_tc = train_bigram_counter[test_bigram]
    if smoothing:
        w1_count += 1.0
        gram_tc += 1.0
        return float(gram_tc) / float(w1_count)
    else:  # avoid division errors
        if w1_count == 0.0:
            return 0.0
        return gram_tc / float(w1_count)


# trigram (w1, w2, w3)
# tc = count in training data
def trigram_probability(test_trigram, train_trigram_counter, train_bigram_counter, smoothing=True):
    """Returns a float calculating the probability for an individual 3-gram.
    Default settings smooths data, set smoothing=False for unsmoothed probability."""
    w1_w2_count = train_bigram_counter[test_trigram[0:2]]
    gram_tc = train_trigram_counter[test_trigram]
    if smoothing:
        w1_w2_count += 1.0
        gram_tc += 1.0
        return float(gram_tc) / float(w1_w2_count)
    else:  # avoid division errors
        if w1_w2_count == 0.0:
            return 0.0
        return gram_tc / float(w1_w2_count)


# fourgram (w1, w2, w3, w4)
# tc = count in training data
def fourgram_probability(test_fourgram, train_fourgram_counter, train_trigram_counter, smoothing=True):
    """Returns a float calculating the probability for an individual 4-gram.
    Default settings smooths data, set smoothing=False for unsmoothed probability."""
    w1_w2_w3_count = train_trigram_counter[test_fourgram[0:3]]
    gram_tc = train_fourgram_counter[test_fourgram]
    if smoothing:
        w1_w2_w3_count += 1.0
        gram_tc += 1.0
        return float(gram_tc) / float(w1_w2_w3_count)
    else:  # avoid division errors
        if w1_w2_w3_count == 0:
            return 0.0
        return float(gram_tc) / float(w1_w2_w3_count)


# Not found counts
not_found_2gram = 66736
not_found_3gram = 146983
not_found_4gram = 185093


def log_likelihood_w_p(test_ngram_counter, train_n_counter, train_n_1_counter, n, smoothing):
    """
    Returns three items: avg log likelihood (float), words not found (int) - 0 if smoothed, and perplexity (float).
    test_ngram_counter = counter object of test data
    train_n_counter = counter of test data
    train_n_1_counter = counter of n-1 gram test data
    n = number of n grams
    smoothing = boolean, true for smoothing, false for no smoothing
    """
    test_ngrams = test_ngram_counter.keys()
    items_not_found = 0  # Items whose probabilities return zero in unsmoothed model, to adjust the average later
    likelihood: float = 0.0
    perplexity: float = 1.0
    p: float = 0.0
    for gram in test_ngrams:
        if n == 2:
            p: float = bigram_probability(gram, train_n_counter, train_n_1_counter, smoothing)
        if n == 3:
            p: float = trigram_probability(gram, train_n_counter, train_n_1_counter, smoothing)
        if n == 4:
            p: float = fourgram_probability(gram, train_n_counter, train_n_1_counter, smoothing)
        if p != 0:
            if smoothing:
                if n == 2:
                    perplexity *= math.pow((1 / float(p)), 1 / len(test_ngrams))
                if n == 3:
                    perplexity *= math.pow((1 / float(p)), 1 / len(test_ngrams))
                if n == 4:
                    perplexity *= math.pow((1 / float(p)), 1 / len(test_ngrams))
            if not smoothing:
                if n == 2:
                    perplexity *= math.pow((1 / float(p)), 1 / (len(test_ngrams) - not_found_2gram))
                if n == 3:
                    perplexity *= math.pow((1 / float(p)), 1 / (len(test_ngrams) - not_found_3gram))
                if n == 4:
                    perplexity *= math.pow((1 / float(p)), 1 / (len(test_ngrams) - not_found_4gram))
            likelihood += math.log(float(p))
        else:
            items_not_found += 1
    return (likelihood / float((len(test_ngrams)) - items_not_found)), items_not_found, perplexity


def sent_probability(test_ngram, train_n_counter, train_n_1_counter, n):
    """
    Returns one float: probability for sentence.
    test_ngram_counter = counter object of test data
    train_n_counter = counter of test data
    train_n_1_counter = counter of n-1 gram test data
    n = number of n grams
    smoothing = boolean, true for smoothing, false for no smoothing
    """
    length_ngrams = len(test_ngram)
    items_not_found = 0  # Items whose probabilities return zero in unsmoothed model, to adjust the average later
    probs = []
    final_probability: float = 1.0
    p: float = 0.0
    for gram in test_ngram:
        if n == 2:
            p: float = bigram_probability(gram, train_n_counter, train_n_1_counter, smoothing=True)
        if n == 3:
            p: float = trigram_probability(gram, train_n_counter, train_n_1_counter, smoothing=True)
        if n == 4:
            p: float = fourgram_probability(gram, train_n_counter, train_n_1_counter, smoothing=True)
        if p != 0:
            probs.append(float(p))
        else:
            items_not_found += 1
    for prob in probs:
        final_probability *= prob
    if (length_ngrams - items_not_found) == 0.0:
        return 0.0
    else:
        return math.log(final_probability)


# Get the average log-likelihoods
s_2gram_ll, a, s_2gram_p = log_likelihood_w_p(count_test_2grams, count_train_2grams, count_train_1grams, 2,
                                              smoothing=True)
s_3gram_11, b, s_3gram_p = log_likelihood_w_p(count_test_3grams, count_train_3grams, count_train_2grams, 3,
                                              smoothing=True)
s_4gram_11, c, s_4gram_p = log_likelihood_w_p(count_test_4grams, count_train_4grams, count_train_3grams, 4,
                                              smoothing=True)
s_2gram_11, not_found_2gram, u_2gram_p = log_likelihood_w_p(count_test_2grams, count_train_2grams,
                                                            count_train_1grams, 2, smoothing=False)
u_3gram_11, not_found_3gram, u_3gram_p = log_likelihood_w_p(count_test_3grams, count_train_3grams,
                                                            count_train_2grams, 3, smoothing=False)
u_4gram_11, not_found_4gram, u_4gram_p = log_likelihood_w_p(count_test_4grams, count_train_4grams,
                                                            count_train_3grams, 4, smoothing=False)

# Create some data about our training set
training_files = count_files('Dataset/TrainingSet/')
training_sentences = count_sentences('Dataset/TrainingSet/')
test_files = count_files('Dataset/TestSet/')
test_sentences = count_sentences('Dataset/TestSet/')

# Find amount of unknown words
training_words = count_train_1grams.keys()
test_words = count_test_1grams.keys()
unknown_words = 0
for word in test_words:
    if word not in training_words:
        unknown_words += 1

# Print an analysis of our data
print("Training Data Analysis:")
print("Total Files: " + str(training_files))
print("Unique Unigrams: " + str(len(count_train_1grams.keys())))
print("Unique Bigrams: " + str(len(count_train_2grams.keys())))
print("Unique Trigrams: " + str(len(count_train_3grams.keys())))
print("Unique 4-grams: " + str(len(count_train_4grams.keys())))
print("Total Sentences: " + str(training_sentences))
print()
print("Test Data Analysis:")
print("Total Files: " + str(test_files))
print("Unique Unigrams: " + str(len(count_test_1grams.keys())))
print("Unique Bigrams: " + str(len(count_test_2grams.keys())))
print("Unique Trigrams: " + str(len(count_test_3grams.keys())))
print("Unique 4-grams: " + str(len(count_test_4grams.keys())))
print("Total Sentences: " + str(test_sentences))
print("Number of unknown words: " + str(unknown_words))
print()
print("Log likelihoods:")
print("Bigram log-likelihood with smoothing:", s_2gram_ll)
print("Trigram log-likelihood with smoothing: ", s_3gram_11)
print("4-gram log-likelihood with smoothing: ", s_4gram_11)
print("Bigram log-likelihood no smoothing:", s_2gram_11)
print("   Bigrams with probability count 0:", not_found_2gram)
print("Trigram log-likelihood no smoothing:", u_3gram_11)
print("   Trigrams with probability count 0:", not_found_3gram)
print("4-gram log-likelihood no smoothing:", u_4gram_11)
print("   4-grams with probability count 0:", not_found_4gram)
print()
print("Bigram perplexity with smoothing:", s_2gram_p)
print("Trigram perplexity with smoothing:", s_3gram_p)
print("4-gram perplexity with smoothing:", s_4gram_p)
print("Bigram perplexity without smoothing:", u_2gram_p)
print("Trigram perplexity without smoothing:", u_3gram_p)
print("4-gram perplexity without smoothing:", u_4gram_p)

# Make some unigram charts
matplotlib.rc('xtick', labelsize=8)
test_freq_dist_1gram.plot(40, title="Frequency Distribution of Top 40 Unigrams in Test Corpus", linewidth=1)
train_freq_dist_1gram.plot(40, title="Frequency Distribution of Top 40 Unigrams in Training Corpus", linewidth=1)


def example_sentence(sentence):
    """Takes a sentence and gives back the probabilities."""
    sent_2gram = list(ngrams(sent_tokenize(sentence, 2), 2))
    sent_3gram = list(ngrams(sent_tokenize(sentence, 3), 3))
    sent_4gram = list(ngrams(sent_tokenize(sentence, 4), 4))
    print()
    print("Sentence: \"" + sentence + "\"")
    print("Bigram probability:", sent_probability(sent_2gram, count_train_2grams, count_train_1grams, 2))
    print("Trigram probability:", sent_probability(sent_3gram, count_train_3grams, count_train_2grams, 3))
    print("4-gram probability:", sent_probability(sent_4gram, count_train_4grams, count_train_3grams, 4))


example_sentence("It also calls for a 10-year plan to restore closed-down rail links and development of harbours.")
example_sentence("More than 60,000 such trees will be recycled this year, according to community service charity CSV")
example_sentence("These trees would once again breathe life into the soil")
