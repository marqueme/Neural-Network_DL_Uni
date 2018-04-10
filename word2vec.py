from gensim.models import word2vec
from gensim import corpora
import sys, getopt
import re
import string
from nltk.corpus import stopwords
import logging

# CONST FOR LABELS
CONST_BACKCHANNEL = 0
CONST_STATEMENT = 1
CONST_QUESTION = 2
CONST_OPINION = 3


# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def categorize_lines(inputfile):
    """
    Extracts the lines from the file and sorts them by category
    :param inputfile: the input file containing the utterances
    :return: a list as :
    [
        [backchannel_line1, backchannel_line2, backchannel_line3,...]
        [statement_line1, statement_line2, statement_line3,...]
        [question_line1, question_line2, question_line3,...]
        [opinion_line1, opinion_line2, opinion_line3,...]
    ]
    """
    utterance_statement = []
    utterance_opinion = []
    utterance_backchannel = []
    utterance_question = []

    sorted_utterances = []

    with open(inputfile) as infile:
        # store the whole line is separate lists
        for line in infile:
            category = get_category(line)
            if category == "backchannel":
                utterance_backchannel.append(line)
            if category == "statement":
                utterance_statement.append(line)
            if category == "question":
                utterance_question.append(line)
            if category == "opinion":
                utterance_opinion.append(line)

    # this order is important and affects the const declarations
    sorted_utterances.append(utterance_backchannel)
    sorted_utterances.append(utterance_statement)
    sorted_utterances.append(utterance_question)
    sorted_utterances.append(utterance_opinion)

    return sorted_utterances


def get_category(line):
    pattern = r'^\d+_\d+(\s+|\t)(\w+)'
    match = re.search(pattern, line)
    if match:
        return match.group(2)


def get_utterance(line):
    pattern = r'^\d+_\d+\s+(\w+)\s+(.*)$'
    # group1 = category
    # group2 = utterance

    match = re.search(pattern, line)
    if match:
        # return the utterance without punctuation and lower-cased
        utterance = match.group(2).lower()
        for c in string.punctuation:
            utterance = utterance.replace(c, "")
        return utterance


def get_id(line):
    pattern = r'(^\d+_\d+)(\s+|\t)'
    # group1 = id

    match = re.search(pattern, line)
    if match:
        id = match.group(1)
        return id


def gen_model(categorized_utterances):
    """
    Generates the models for all utterances in the provided list
    :param categorized_utterances: list of all utterance lines sorted by category.
    :return: list of all models sorted by category in the following shape:
     [
         [(backchannel)
           [utterance id, label, [[word1 vec],[word2 vec],...]]
           [utterance id, label, [[word1 vec],[word2 vec],...]]
           [utterance id, label, [[word1 vec],[word2 vec],...]]
         ],
         [(statement)
           [utterance id, label, [[word1 vec],[word2 vec],...]]
           [utterance id, label, [[word1 vec],[word2 vec],...]]
           [utterance id, label, [[word1 vec],[word2 vec],...]]
         ],
         [...]
     ]
    """
    lines_backchannel = categorized_utterances[CONST_BACKCHANNEL]
    lines_statement = categorized_utterances[CONST_STATEMENT]
    lines_question = categorized_utterances[CONST_QUESTION]
    lines_opinion = categorized_utterances[CONST_OPINION]

    categorized_entries = []  # return list. Append results to it in the same order as categorized_utterances
    entries_backchannel = []
    entries_statement = []
    entries_question = []
    entries_opinion = []

    for line in lines_backchannel:
        entries_backchannel.append(gen_entry(line, CONST_BACKCHANNEL))
    for line in lines_statement:
        entries_statement.append(gen_entry(line, CONST_STATEMENT))
    for line in lines_question:
        entries_question.append(gen_entry(line, CONST_QUESTION))
    for line in lines_opinion:
        entries_opinion.append(gen_entry(line, CONST_OPINION))

    # this order is important and matches the const declarations
    categorized_entries.append(entries_backchannel)
    categorized_entries.append(entries_statement)
    categorized_entries.append(entries_question)
    categorized_entries.append(entries_opinion)

    return categorized_entries


def gen_word2vec(utterance):
    num_features = 300
    min_word_count = 1
    num_workers = 4
    context = 10

    utterance = [utterance.split()]

    try:
        model = word2vec.Word2Vec(
            utterance,
            workers=num_workers,
            size=num_features,
            min_count=min_word_count,
            window=context)

        #print "Resulting keys for '" + str(utterance) + "'"
        #print "  " + str(list(model.vocab.keys()))
        #print " " + str(list(model.wv.vocab.keys()))
        return model
    except Exception as e:
        print "Problem with utterance: " + utterance

    return null

def gen_entry(line, label):
    """
    Generates a list of the following structure: [utterance id, label, [[word1 vec],[word2 vec],...]]
    :param label: the label identifying the category of the utterance
    :param line: a line of the input file, as read.
    :return: list
    """
    id = get_id(line)
    utterance = get_utterance(line)
    utterance_model = gen_word2vec(utterance)  # model for the whole utterance
    word_vectors = []

    # go through the model and extract each word's vector
    if len(utterance.split()) > 1:
        for word in utterance.split():
            word_vector = utterance_model[word].tolist()
            word_vectors.append(word_vector)
    else:
        word_vector = utterance_model[utterance].tolist()
        word_vectors.append(word_vector)

    entry = [id, label, word_vectors]

    #print "Entry generated for '" + line + "':  " + str(entry)

    return entry


def main(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile="])
    except getopt.GetoptError:
        print 'Run the command as: word2vec.py -i <inputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg

    categorized_utterances = categorize_lines(inputfile)

    categorized_entries = gen_model(categorized_utterances)

    # write to outputfile. Each line of the file is an entry
    outputfile = "output.txt"
    with open(outputfile, 'w') as f:
        # go through the main list of lists
        for category in categorized_entries:
            for entry in category:
                f.write(str(entry) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
