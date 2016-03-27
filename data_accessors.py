#coding: utf-8

""" This module contains some data and funcs used to test the gramdicmaker system.
Here are the variables defining the paradigms involved in a study conducted:
UDMURT_RELEVANT: a tuple of paradigms processed with a module (covered with category description);
ALBANIAN_RELEVANT_I, ALBANIAN_RELEVANT_II: paradigms included into the two data sets.
The tuple containing all the paradigms described is a sum of these two tuples;
KATHAREVOUSA_RELEVANT: paradigms covered with katharevousa category description.

"""

import codecs

UDMURT_RELEVANT = (
    u'connect_adjectives',
    u'connect_verbs-I',
    u'Noun-num-j',
    u'Noun-num-soft',
    u'Noun-case',
    u'connect_verbs-I-soft',
    u'Noun-num-vowel',
    u'Noun-num-ija',
    u'connect_verbs-II',
    u'Noun-num-consonant'
)

ALBANIAN_RELEVANT_II = (
    u'adj-1-5',
    u'adj-2-1-a',
    u'adj-2-1',
    u'adj-2-2-a',
    u'adj-2-2',
    u'adj-2-3',
    u'adj-3-1',
    u'adj-3-2'
)

ALBANIAN_RELEVANT_I = (
    u'adj-1-1-a',
    u'adj-1-1-b',
    u'adj-1-1',
    u'adj-1-2-a',
    u'adj-1-2',
    u'adj-1-3',
    u'adj-1-4-a',
    u'adj-1-4'
)

KATHAREVOUSA_RELEVANT = (u'adj_os1', )


class LangTestData(object):
    """ A set of methods to quickly access the data. All the public  methods return a triple:
    (a list of wordforms, path to paradigm description, path to conversion description).

    """

    @staticmethod
    def _tsv_reader(filename):
        testWordforms = []
        with codecs.open(filename, 'r', 'utf-8-sig') as f:
            for l in f:
                try:
                    line = l.split()
                    counter = int(line[1].strip())
                    line = line[0].strip()
                    for i in range(counter):
                        testWordforms.append(line)
                except:
                    pass
        return testWordforms

    @staticmethod
    def _colon_reader(filename):
        testWordforms = []
        with codecs.open(filename, 'r', 'utf-8-sig') as f:
            for l in f:
                try:
                    line = l.split()
                    counter = int(line[1].strip())
                    line = line[0].strip().strip(u':')
                    for i in range(counter):
                        testWordforms.append(line)
                except:
                    pass
        return testWordforms

    @staticmethod
    def katharevousa():
        """ Read test wordform file for katharevousa and paths to the grammar data.
        Return tuple: (list of all the wordforms, paradigm path, conversion path).

        """
        testWordforms = LangTestData._tsv_reader(u'test_data/katharevousa/types_without_circumflexes.txt')
        paradigms = u'test_data/katharevousa/paradigms_katharevousa_incomplete.txt'
        conv = u'test_data/katharevousa/stems_conversion.txt'
        return testWordforms, paradigms, conv

    @staticmethod
    def kazakh_nouns():
        """ Read test wordform file for kazakh and paths to the grammar data.
        Return tuple: (list of all the wordforms, noun paradigm path, conversion path).

        """
        testWordforms = LangTestData._colon_reader(u'test_data/kazakh/concordance.txt')
        paradigms = u'test_data/kazakh/paradigms_N.txt'
        conv = u'test_data/kazakh/stems_conversion_N.txt'
        return testWordforms, paradigms, conv

    @staticmethod
    def kazakh_verbs():
        """ Read test wordform file for kazakh and paths to the grammar data.
        Return tuple: (list of all the word forms, verb paradigm path, conversion path).

        """
        testWordforms = LangTestData._colon_reader(u'test_data/kazakh/concordance.txt')
        paradigms = u'test_data/kazakh/paradigms_V.txt'
        conv = u''
        return testWordforms, paradigms, conv

    @staticmethod
    def kazakh():
        """ Read test wordform file for kazakh and paths to the grammar data.
        Return tuple: (list of all the word forms, verb paradigm path, conversion path).

        """
        testWordforms = LangTestData._colon_reader(u'test_data/kazakh/concordance.txt')
        paradigms = u'test_data/kazakh/paradigms.txt'
        conv = u'test_data/kazakh/stems_conversion_N.txt'
        return testWordforms, paradigms, conv

    @staticmethod
    def greek():
        testWordforms = LangTestData._tsv_reader(u'test_data/greek/greek_concordance_2014.02.txt')
        paradigms = u'test_data/greek/paradigms.txt'
        conv = u''
        return testWordforms, paradigms, conv

    @staticmethod
    def albanian():
        testWordforms = LangTestData._colon_reader(u'test_data/albanian/concordance_freq.txt')
        paradigms = u'test_data/albanian/paradigms-all.txt'
        conv = u''
        return testWordforms, paradigms, conv

    @staticmethod
    def udmurt():
        testWordforms = LangTestData._tsv_reader(u'test_data/udmurt/concNormal.csv')
        paradigms = u'test_data/udmurt/paradigms.txt'
        conv = u''
        return testWordforms, paradigms, conv