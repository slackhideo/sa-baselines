import os
import re
import codecs
import numpy as np
import pandas as pd
#import contractions
from sklearn.svm import SVC
#from stop_words import read_stop_words
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer


DATA_BASE_DIR = 'datasets'


script_base_dir = os.path.dirname(os.path.realpath(__file__))


def read_movie_reviews(data_dir='movie_reviews'):
    with codecs.open(os.path.join(script_base_dir,
                                  DATA_BASE_DIR,
                                  data_dir,
                                  'mr-polarity.neg'),
                     'r',
                     encoding='utf-8',
                     errors='ignore') as f:
        Xn_text = f.read().splitlines()
    with codecs.open(os.path.join(script_base_dir,
                                  DATA_BASE_DIR,
                                  data_dir,
                                  'mr-polarity.pos'),
                     'r',
                     encoding='utf-8',
                     errors='ignore') as f:
        Xp_text = f.read().splitlines()

    X = []
    X.extend(Xn_text)
    X.extend(Xp_text)

    yn = np.zeros(len(Xn_text), dtype=np.int64)
    yp = np.ones (len(Xp_text), dtype=np.int64)
    y  = np.append(yn, yp)

    return X, y


def read_customer_reviews(data_dir='customer_reviews'):
    positives = []
    negatives = []
    mixed = []

    for filename in sorted(os.listdir(os.path.join(script_base_dir,
                                                   DATA_BASE_DIR,
                                                   data_dir))):
        with codecs.open(os.path.join(script_base_dir,
                                      DATA_BASE_DIR,
                                      data_dir,
                                      filename),
                         'r',
                         encoding='utf-8',
                         errors='ignore') as f:
            for l in f:
                if re.search(r"\[\+.\]", l) is not None:
                    if re.search(r"\[\-.\]", l) is not None:
                        mixed.append(l[(l.index('##') + 2):].strip())
                    else:
                        positives.append(l[(l.index('##') + 2):].strip())
                elif re.search(r"\[\-.\]", l) is not None:
                    negatives.append(l[(l.index('##') + 2):].strip())

    X = []
    X.extend(negatives)
    X.extend(positives)

    yn = np.zeros(len(negatives), dtype=np.int64)
    yp = np.ones (len(positives), dtype=np.int64)
    y  = np.append(yn, yp)

    return X, y


def read_mpqa(data_dir='mpqa'):
    positives = []
    negatives = []

    doclist_files = ['doclist.mpqaOriginalSubset',
                     'doclist.opqaSubset',
                     'doclist.ula-luSubset',
                     'doclist.ulaSubset',
                     'doclist.xbankSubset']

    for doclist in doclist_files:
        with open(os.path.join(script_base_dir, DATA_BASE_DIR, data_dir, doclist), 'r') as opqa_files:
            for opqa_file_path in opqa_files:
                with open(os.path.join(script_base_dir,
                                       DATA_BASE_DIR,
                                       data_dir,
                                       'man_anns',
                                       opqa_file_path.strip(),
                                       'gateman.mpqa.lre.2.0'),
                          'r') as annotation_file:
                    with codecs.open(os.path.join(script_base_dir,
                                                  DATA_BASE_DIR,
                                                  data_dir,
                                                  'docs',
                                                  opqa_file_path.strip()),
                                     'r',
                                     encoding='utf-8',
                                     errors='ignore') as doc:
                        for l in annotation_file:
                            if not l.lstrip().startswith('#'):
                                fields = l.split('\t')
                                assert len(fields) == 5
                                if fields[3] == 'GATE_expressive-subjectivity':
                                    intensity = re.search(r'intensity\s*=\s*"(.*?)"', fields[4])
                                    if intensity is not None and intensity.group(1) not in ['low', 'neutral']:
                                        # Subjective phrase
                                        polarity = re.search(r'polarity\s*=\s*"(.*?)"', fields[4])
                                        if polarity is None or not polarity.group(1):
                                            continue
                                        span = fields[1].split(',')
                                        assert len(span) == 2
                                        # Do not consider empty strings
                                        if span[0] == span[1]:
                                            continue
                                        doc.seek(int(span[0]))
                                        phrase = re.sub(r'\s+', ' ', doc.read(int(span[1]) - int(span[0])))
                                        if polarity.group(1) == 'positive':
                                            positives.append(phrase)
                                        elif polarity.group(1) == 'negative':
                                            negatives.append(phrase)
                                        continue

                                # Other condition for subjectivity
                                elif fields[3] == 'GATE_direct-subjective':
                                    intensity = re.search(r"intensity\s*=\s*\"(.*?)\"", fields[4])
                                    if intensity is not None and intensity.group(1) not in ['low', 'neutral']:
                                        insubstantial = re.search(r'insubstantial\s*=\s*"(.*?)"', fields[4])
                                        if insubstantial is None:
                                            # Subjective phrase
                                            polarity = re.search(r'polarity\s*=\s*"(.*?)"', fields[4])
                                            if polarity is None or not polarity.group(1):
                                                continue
                                            span = fields[1].split(',')
                                            assert len(span) == 2
                                            # Do not consider empty strings
                                            if span[0] == span[1]:
                                                continue
                                            doc.seek(int(span[0]))
                                            phrase = re.sub(r'\s+', ' ', doc.read(int(span[1]) - int(span[0])))
                                            if polarity.group(1) == 'positive':
                                                positives.append(phrase)
                                            elif polarity.group(1) == 'negative':
                                                negatives.append(phrase)

    X = []
    X.extend(negatives)
    X.extend(positives)

    yn = np.zeros(len(negatives), dtype=np.int64)
    yp = np.ones (len(positives), dtype=np.int64)
    y  = np.append(yn, yp)

    return X, y


def read_yelp(data_dir='yelp_2015_v2_binary'):

    # Training instances
    data = pd.read_csv(os.path.join(script_base_dir,
                                    DATA_BASE_DIR,
                                    data_dir,
                                    'train_2perc.csv'),
                       header=None,
                       names=['class', 'text'])
    X_train = [re.sub(r'\\n', ' ', text) for text in data['text'].values]
    y_train = data['class'].values - 1 # Transforms the classes into '0' and '1'

    # Test instances
    data = pd.read_csv(os.path.join(script_base_dir,
                                    DATA_BASE_DIR,
                                    data_dir,
                                    'test_2perc.csv'),
                       header=None,
                       names=['class', 'text'])
    X_test = [re.sub(r'\\n', ' ', text) for text in data['text'].values]
    y_test = data['class'].values - 1 # Transforms the classes into '0' and '1'

    X = []
    X.extend(X_train)
    X.extend(X_test)

    bow = CountVectorizer()
    X_bow = bow.fit_transform(X)

    X_bow_train = X_bow[:len(X_train)]
    X_bow_test  = X_bow[len(X_train):]

    return X_bow_train, X_bow_test, y_train, y_test


#def fix_contractions(X):
#    fixed = []
#
#    for l in X:
#        fixed.append(contractions.fix(l))
#
#    return fixed


def pre_process(X, y):
#    stop_words = read_stop_words('stop_words_minimal.txt')

#    bow = CountVectorizer(stop_words=stop_words)
    bow = CountVectorizer(stop_words=None)
    X_bow = bow.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_bow,
                                                        y,
                                                        test_size=0.1,
                                                        shuffle=True,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def cross_validation(model, parameters, X_train, X_test, y_train, y_test):
    clf = GridSearchCV(model, parameters, cv=10, refit=True)
    clf.fit(X_train, y_train)

    print("Best result:")
    print()
    print("{0:0.3f} for {1}".format(clf.best_score_, clf.best_params_))
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds  = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()
    print("Detailed report:")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    return clf.best_estimator_


def dataset_stats(X, y):
    assert len(X) == len(y)

    # Number of instances
    n = len(X)

    d = dict(zip(*np.unique(y, return_counts=True)))

    # Number of negative and positive instances
    n_neg = d[0]
    n_pos = d[1]

    num_words = []

    for sent in X:
        num_words.append(len(sent.split()))

    # Average number of words per instance
    w = np.mean(num_words)

    bow = CountVectorizer()
    X_bow = bow.fit_transform(X)

    # Size of the vocabulary
    v = X_bow.shape[1]

    print('N = {0}, N+ = {1}, N- = {2}, w = {3:0.2f}, V = {4}'.format(n,
                                                                      n_pos,
                                                                      n_neg,
                                                                      w,
                                                                      v))


def logistic_regression(X_train, X_test, y_train, y_test):

    # Logistic Regression with built-in Cross-Validation
    print("### Logistic Regression Cross-Validation ###")
    model = LogisticRegressionCV(Cs=[0.1, 1, 10, 100, 1000],
                                 cv=10,
                                 refit=True,
                                 random_state=42,
                                 max_iter=500)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("Accuracy = {0:0.3f}\n".format(score))


#    # Logistic Regression
#    print("### Logistic Regression ###")
#    model = LogisticRegression(random_state=42, max_iter=500)
#
#    parameters = [{'penalty': ['none', 'l1', 'l2'],
#                   'C': [0.1, 1, 10, 100, 1000],
#                   'solver': ['saga']},
#                  {'penalty': ['none', 'l2'],
#                   'C': [0.1, 1, 10, 100, 1000],
#                   'solver': ['newton-cg', 'lbfgs', 'sag']},
#                  {'penalty': ['l1', 'l2'],
#                   'C': [0.1, 1, 10, 100, 1000],
#                   'solver': ['liblinear']}]
#
#    best = cross_validation(model, parameters, X_train, X_test, y_train, y_test)
#    score = best.score(X_test, y_test)
#    print("Accuracy = {0:0.3f}\n".format(score))


def svm(X_train, X_test, y_train, y_test):

    # Linear SVM
    print("### Linear SVM ###")
    model = SVC(random_state=42, kernel='linear')

    parameters = [{'C': [0.01, 0.1, 1, 10, 100, 1000]}]

    best = cross_validation(model, parameters, X_train, X_test, y_train, y_test)
    score = best.score(X_test, y_test)
    print("Accuracy = {0:0.3f}\n".format(score))


    # SVM with RBF kernel
    print("### SVM with RBF kernel ###")
    model = SVC(random_state=42, kernel='rbf')

    parameters = [{'C': [1, 10, 100, 1000],
                   'gamma': ['auto', 'scale', 0.001, 0.001, 0.01, 0.1]}]

    best = cross_validation(model, parameters, X_train, X_test, y_train, y_test)
    score = best.score(X_test, y_test)
    print("Accuracy = {0:0.3f}\n".format(score))


def naive_bayes(X_train, X_test, y_train, y_test):

    # Bernoulli Naïve Bayes
    print("### Bernoulli Naïve Bayes ###")
    model = BernoulliNB()

    parameters = [{'alpha': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]}]

    best = cross_validation(model, parameters, X_train.toarray(), X_test.toarray(), y_train, y_test)
    score = best.score(X_test.toarray(), y_test)
    print("Accuracy = {0:0.3f}\n".format(score))


    # Multinomial Naïve Bayes
    print("### Multinomial Naïve Bayes ###")
    model = MultinomialNB()

    parameters = [{'alpha': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]}]

    best = cross_validation(model, parameters, X_train, X_test, y_train, y_test)
    score = best.score(X_test, y_test)
    print("Accuracy = {0:0.3f}\n".format(score))


def random_forest(X_train, X_test, y_train, y_test):

    # Random Forest
    print("### Random Forest ###")
    model = RandomForestClassifier(random_state=42)

    parameters = [{'n_estimators': [5, 10, 50, 100, 500, 1000],
                   'max_depth': [5, 10, None],
                   'max_features': [1, 2, 'auto', 'sqrt', 'log2', None],
                   'class_weight': ['balanced', None]}]

    best = cross_validation(model, parameters, X_train, X_test, y_train, y_test)
    score = best.score(X_test, y_test)
    print("Accuracy = {0:0.3f}\n".format(score))


def main():
    mr_X, mr_y = read_movie_reviews()
#    mr_X = fix_contractions(mr_X)
    dataset_stats(mr_X, mr_y)
    mr_X_tr, mr_X_te, mr_y_tr, mr_y_te = pre_process(mr_X, mr_y)

    cr_X, cr_y = read_customer_reviews()
#    cr_X = fix_contractions(cr_X)
    dataset_stats(cr_X, cr_y)
    cr_X_tr, cr_X_te, cr_y_tr, cr_y_te = pre_process(cr_X, cr_y)

    mpqa_X, mpqa_y = read_mpqa()
#    mpqa_X = fix_contractions(mpqa_X)
    dataset_stats(mpqa_X, mpqa_y)
    mpqa_X_tr, mpqa_X_te, mpqa_y_tr, mpqa_y_te = pre_process(mpqa_X, mpqa_y)

    yelp_X_tr, yelp_X_te, yelp_y_tr, yelp_y_te = read_yelp()


    logistic_regression(mr_X_tr, mr_X_te, mr_y_tr, mr_y_te)
    logistic_regression(cr_X_tr, cr_X_te, cr_y_tr, cr_y_te)
    logistic_regression(mpqa_X_tr, mpqa_X_te, mpqa_y_tr, mpqa_y_te)
    logistic_regression(yelp_X_tr, yelp_X_te, yelp_y_tr, yelp_y_te)

    svm(mr_X_tr, mr_X_te, mr_y_tr, mr_y_te)
    svm(cr_X_tr, cr_X_te, cr_y_tr, cr_y_te)
    svm(mpqa_X_tr, mpqa_X_te, mpqa_y_tr, mpqa_y_te)
    svm(yelp_X_tr, yelp_X_te, yelp_y_tr, yelp_y_te)

    naive_bayes(mr_X_tr, mr_X_te, mr_y_tr, mr_y_te)
    naive_bayes(cr_X_tr, cr_X_te, cr_y_tr, cr_y_te)
    naive_bayes(mpqa_X_tr, mpqa_X_te, mpqa_y_tr, mpqa_y_te)
    naive_bayes(yelp_X_tr, yelp_X_te, yelp_y_tr, yelp_y_te)

    random_forest(mr_X_tr, mr_X_te, mr_y_tr, mr_y_te)
    random_forest(cr_X_tr, cr_X_te, cr_y_tr, cr_y_te)
    random_forest(mpqa_X_tr, mpqa_X_te, mpqa_y_tr, mpqa_y_te)
    random_forest(yelp_X_tr, yelp_X_te, yelp_y_tr, yelp_y_te)


if __name__ == "__main__":
    main()
