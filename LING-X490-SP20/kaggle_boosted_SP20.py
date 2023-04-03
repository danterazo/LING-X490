# LING-X 490 FA19 Final: Boosted Kaggle SVM
# Dante Razo, drazo

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from string import capwords
import sklearn.metrics
import pandas as pd

pd.options.mode.chained_assignment = None  # suppress SettingWithCopyWarning


def get_data(verbose, boost_threshold, sample_types, sample_size=15000):
    data_dir = "../data/kaggle_data"  # common directory for all datasets
    dataset = "train.target+comments.tsv"  # 'test' for classification problem
    print(f"Importing `{dataset}`...") if verbose else None  # progress indicator
    data_list = []  # temporary; used for constructing dataframe

    # import data
    with open(f"{data_dir}/{dataset}", "r", encoding="utf-8") as d:
        entries = d.readlines()

        for e in entries:
            splitLine = e.split("\t", 1)

            if len(splitLine) is 2:  # else: there's no score, so throw the example out
                data_list.append([float(splitLine[0]), splitLine[1]])
            else:
                print(f"error: {splitLine}")  # debugging

    data = pd.DataFrame(data_list, columns=["score", "comment_text"])
    print(f"Data {data.shape} imported!") if verbose else None  # progress indicator

    kaggle_threshold = 0.50  # from Kaggle documentation (see page)
    dev = True  # set to FALSE when its time to validate `train` dataset
    shuffle = True  # self-explanatory
    to_return = []  # this function returns a list of lists. Each inner list contains `X` and `y`

    # create class vector
    data["class"] = 0
    data.loc[data.score >= kaggle_threshold, "class"] = 1

    # sampled datasets
    data_len = len(data)
    if sample_size > data_len or sample_size < 1:
        sample_size = data_len  # bound

    boosted_data = boost_data(data[0:sample_size], boost_threshold, verbose)  # TODO: reimplement
    random_sample = data.sample(frac=1).sample(len(data))[0:sample_size]  # shuffle first, then pick out `n` entries

    for s in sample_types:
        if s is "boosted":
            data = boosted_data.sample(frac=1)  # reshuffle
            pass  # debugging, to remove
        elif s is "random":
            data = random_sample.sample(frac=1)  # reshuffle

        X = data.loc[:, data.columns != "class"]
        y = data.loc[:, data.columns == "class"]

        # train: 60%, dev: 20%, test: 20%
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                    test_size=0.2,
                                                                                    shuffle=shuffle,
                                                                                    random_state=42)

        X_train, X_dev, y_train, y_dev = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                  test_size=0.25,
                                                                                  shuffle=shuffle,
                                                                                  random_state=42)

        to_return.append([X_train, X_dev, y_train, y_dev]) if dev \
            else to_return.append([X_train, X_test, y_train, y_test])  # use dev sets if dev=TRUE

    # [[boosted X, y], [randomly sampled X, y]]
    return to_return


# boosting; filters on abusive language
def boost_data(data, boost_threshold, verbose):
    print(f"Boosting data...") if verbose else None

    # hate speech lexicon tasks
    # import
    lexicon_dir = "../Data/kaggle_data/lexicon_wiegand"
    version = "base"  # or "expanded"
    df = pd.read_csv(f"{lexicon_dir}/{version}Lexicon.txt", sep='\t', header=None)
    lexicon = pd.DataFrame(columns=["word", "part", "hate"])

    # split into three features
    lexicon[["word", "part"]] = df[0].str.split('_', expand=True)
    lexicon["hate"] = df[1]

    # create list of abusive words
    hate = list(lexicon[lexicon["hate"]]["word"])

    """ save abusive lexicon to CSV
    # lexicon[lexicon["hate"]]["word"].to_csv('hate_lexicon.csv', index=False)
    """

    filtered_data = topic_filter(data, hate, verbose)
    return filtered_data


def topic_filter(data, hate_lexicon, verbose):
    # source (built upon): https://dictionary.cambridge.org/us/topics/religion/islam/d
    islam_wordbank = ["allah", "caliphate", "fatwa", "hadj", "hajj", "halal", "headscarf", "hegira", "hejira", "hijab",
                      "islam", "islamic", "jihad", "jihadi", "mecca", "minaret", "mohammeden", "mosque", "muhammad",
                      "mujahideen", "muslim", "prayer", "mat", "prophet", "purdah", "ramadan", "salaam", "sehri",
                      "sharia", "shia", "sunni", "shiism", "sufic", "sufism", "suhoor", "sunna", "koran", "qur'an",
                      "yashmak", "ISIS", "ISIL", "al-Qaeda", "Taliban"]

    # TODO: see Sandra's email for suggestions

    # source: https://www.usatoday.com/story/news/2017/03/16/feminism-glossary-lexicon-language/99120600/
    metoo_wordbank = ["metoo", "feminism", "victim", "consent", "patriarchy", "sexism", "misogyny", "misandry",
                      "misogynoir", "cisgender", "transgender", "transphobia", "transmisogyny",
                      "terf", "swef", "non-binary", "woc", "victim-blaming", "trigger", "privilege", "mansplain",
                      "mansplaining", "manspread", "manspreading", "woke", "feminazi"]

    # source: https://en.wikipedia.org/wiki/Wikipedia:List_of_controversial_issues#Politics_and_economics
    politics_wordbank = ["republican", "GOP", "democrats", "liberal", "liberals", "abortion", "brexit",
                         "anti-semitism", "atheism", "conservatives", "capitalism", "communism", "Cuba",
                         "fascism", "Fox News", "immigration", "kashmir", "harambe", "israel", "hitler", "mexico",
                         "neoconservatism", "neoliberalism", "palestine", "9/11", "socialism", "Clinton", "Trump",
                         "Sanders", "Guantanamo", "torture", "Flight 77", "Marijuana", "sandinistas"]

    # source: https://en.wikipedia.org/wiki/Wikipedia:List_of_controversial_issues#History
    history_wordbank = ["Apartheid", "Nazi", "Black Panthers", "Rwandan Genocide", "Jim Crow", "Ku Klux Klan"]

    # source: https://en.wikipedia.org/wiki/Wikipedia:List_of_controversial_issues#Religion
    religion_wordbank = ["jew", "judaism", "christian", "christianity", "Jesus Christ", "Baptist", "WASP", "Protestant",
                         "Westboro Baptist Church"]

    # source: Sandra's suggestions, email from 2020-03-16
    sandra_wordbank = ["trump", "obama", "trudeau", "clinton", "hillary", "donald", "tax", "taxpayer", "vote", "voting",
                       "election", "party", "president", "politician", "women", "woman", "fact", "military", "citizen",
                       "nation", "church", "christian", "muslim", "liberal", "democrat", "republican", "religion",
                       "religious", "administration", "immigrant", "gun", "science", "freedom", "solution",
                       "corporate"]

    # words with special capitalization rules; except from capwords() function call below
    special_caps = ["al-Qaeda", "CNN", "KKK", "LGBT", "LGBTQ", "LGBTQIA"]

    # manually observed abusive words in explicit examples
    explicitly_abusive = ["sh*tty"]

    # future: https://thebestschools.org/magazine/controversial-topics-research-starter/

    # combine the above topics
    combined_topics = islam_wordbank + metoo_wordbank + politics_wordbank + history_wordbank + religion_wordbank + \
                      sandra_wordbank + special_caps + explicitly_abusive

    topic = combined_topics  # easy toggle if you want to focus on a specific topic instead
    topic = list(dict.fromkeys(topic))  # remove dupes

    wordbank = [t.lower() for t in topic]  # lowercase all in topic[]...
    wordbank = wordbank + [capwords(w) for w in wordbank] + special_caps  # ...add capitalized versions...
    wordbank = wordbank + ["#" + word for word in topic]  # ...then add hashtags for all words
    wordbank = list(dict.fromkeys(wordbank))  # remove dupes again cause once isn't enough for some reason
    wordbank_regex = "|".join(wordbank)  # form "regex" string

    # idea: .find() for count. useful for threshold
    topic_data = data[data["comment_text"].str.contains(wordbank_regex)]  # boost data; TODO: redo this

    """ Save topic_data for Brooklyn:
    topic_data.to_csv('filtered_data.csv', index=True)
    print(f"`topic_data` ({len(topic_data)}) saved!")
    """

    return topic_data


""" CONFIGURATION """
mode = "boost_test"  # presets switch: "quick" / "boost_test" / "nohup" / "user"
verbose = True  # print statement flag
sample_type = ["boosted", "random"]  # do both types of sampling

if mode is "quick":  # for development. quick fits
    print("DEVELOPMENT MODE ----------------------")
    analyzer, ngram_upper_bound, sample_size, boost_threshold = ["word", [3], 1000, 1]
    sample_type = ["random"]

elif mode is "boost_test":  # for boosting development
    print("BOOSTING DEVELOPMENT MODE ----------------------")
    analyzer, ngram_upper_bound, sample_size, boost_threshold = ["word", [3], 15000, 1]
    sample_type = ["boosted"]

elif mode is "nohup":  # nohup mode. hard-code inputs here, switch the mode above, then run!
    print("NOHUP MODE -------------------------")
    analyzer = "word"
    ngram_upper_bound = [3]
    sample_size = 50000  # try: 50000. max: 1804874
    boost_threshold = 1
    verbose = False

else:  # user-interactive mode. Good for running locally... not good for nohup
    print("COUNTVECTORIZER CONFIG\n----------------------")
    analyzer = input("Please enter analyzer: ")
    ngram_upper_bound = input("Please enter ngram upper bound(s): ").split()
    sample_size = input("Please enter sample size (< 66839): ")
    boost_threshold = input("Please enter the hate speech threshold: ")  # num of abusive words each entry must contain

""" MAIN """
data = get_data(verbose, boost_threshold, sample_type, sample_size)  # array of data. [[boosted X,y], [random X,y]]

# allows different ngram bounds w/o `Pipeline`
for i in ngram_upper_bound:

    # Try the current parameters with each sampling type
    for t in range(0, len(sample_type)):
        X_train, X_test, y_train, y_test = data[t]

        # Feature engineering: Vectorizer. ML models need features, not just whole tweets
        vec = CountVectorizer(analyzer="word", ngram_range=(1, 1))
        print(f"\nFitting {sample_type[t].capitalize()}-sample CV...") if verbose else None
        X_train_CV = vec.fit_transform(X_train["comment_text"])
        X_test_CV = vec.transform(X_test["comment_text"])

        # Fitting the model
        print(f"Training {sample_type[t].capitalize()}-sample SVM...") if verbose else None
        svm_model = SVC(kernel="linear", gamma="auto")
        svm_params = {'C': [0.1, 1, 10, 100, 1000],  # regularization param
                      'gamma': ["scale", "auto", 1, 0.1, 0.01, 0.001, 0.0001],  # kernel coefficient (R, P, S)
                      'kernel': ["linear", "poly", "rbf", "sigmoid"]}  # SVM kernel (precomputed not supported)
        svm_gs = GridSearchCV(svm_model, svm_params, n_jobs=4, cv=5)
        svm_gs.fit(X_train_CV, y_train.values.ravel())
        print(f"Training complete.") if verbose else None

        # Testing + results
        nl = "" if mode is "nohup" else "\n"  # groups results together when training
        print(f"{nl}Classification Report [{sample_type[t].lower()}, {analyzer}, ngram_range(1,{i})]:\n "
              f"{classification_report(y_test, svm_gs.predict(X_test_CV), digits=6)}")

""" RESULTS & DOCUMENTATION
# KERNEL TESTING (RANDOM, size=50000, gamma="auto", analyzer=word, ngram_range(1,3))
linear:  
rbf:     
poly:    
sigmoid: 
precomputed: N/A, not supported

# BOOSTED CountVectorizer PARAM TESTING (kernel="linear")
word, ngram_range(1,3): 
char, ngram_range(1,3):  

# RANDOM CountVectorizer PARAM TESTING (kernel="linear")
word, ngram_range(1,3):  
char, ngram_range(1,3):  

## Train start (all): 
## Train end (word):  
## Train kill (all):  
"""
