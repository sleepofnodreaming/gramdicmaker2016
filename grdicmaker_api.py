#!/usr/local/bin/python
# coding: utf-8

import codecs
import json
import logging
import os
import random
import re
import tempfile
import time
import shutil

from automaton import FlexAutomaton
from feature_extractors import Thresholds, FeatureExtractor
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import svm, preprocessing
from uniparser.grammar import Grammar  # and this is the module from the parser


STANDARD_THRESHOLD = 0


logging.basicConfig(level=logging.DEBUG, format = u'[%(asctime)s] %(levelname)s: %(message)s')


class _PrefixFinder(FlexAutomaton):
    """
    An automaton to extract a set of prefixes for a paradigm's inflexion.
    >>> obj = _PrefixFinder()
    >>> obj.add(u'.a')
    >>> obj.add(u'.aba')
    >>> obj.add(u'.ba')
    >>> sorted(list(obj.check(u'.ba')))
    [u'.a']
    >>> sorted(list(obj.check(u'.aba')))
    [u'.a', u'.ba']
    """
    _replace = re.compile(ur'[\.#]+', flags=re.U | re.I)

    def __init__(self):
        PREFIX_AUTOMATON_MARKER = u'#'
        self.contents = {}

        FlexAutomaton.__init__(self, empty=PREFIX_AUTOMATON_MARKER)

    def add_paradigm(self, paradigmObject):

        for flex in paradigmObject.flex:
            flexText = flex.comparison_pattern()
            if flexText != self.marker:  # todo find out what to do if an inflection is empty.
                #  In fact, this is useless as a pattern here is not a dot but a number sign
                self.add(flexText)
                # self.fCounter += 1

    def add(self, flex):
        """
        Add all the inflection options to the data structure. As we propose that some dots missing,
        all the options of dot positions should be added.

        """
        patterns = [u'%s', u'%s.', u'.%s', u'.%s.']
        for ptrn in patterns:
            flexFormatted = self._replace.sub(self.marker, ptrn % flex)  # in a pattern flex,
            # add missing dots and replace them all with a number sign
            self.contents[flexFormatted] = flex  # inside the keys, a number sign only
            # is used as an empty transition marker. However, values look like a normal flex.
            FlexAutomaton.add(self, flexFormatted)

    def check(self, flex):
        """
        Given a flex, return a set of possible prefixes.

        """
        options = self.parse(flex)
        relevantPrefixes = set()
        for foundFstFlex, foundScdFlex in options:
            if foundFstFlex != u'#':
                # the second flex found should not be checked whether it exists as there are cases
                # when the remainder consists of valid parts but is not a valid one itself.
                # foundScdFlex = self._replace.sub(u'#', foundScdFlex)
                # if foundScdFlex in self.contents and self._replace.sub(u'.', foundFstFlex) != flex:
                if self._replace.sub(u'.', foundFstFlex) != flex:
                    relevantPrefixes.add(self.contents[foundFstFlex])
        return relevantPrefixes


class _ParadigmFeaturedAutomaton(FlexAutomaton):
    def __init__(self, agglutinative):

        MARKER_DOT = u'.'
        FlexAutomaton.__init__(self, empty=MARKER_DOT)

        self.paradigms = {}
        self.pMetrics = {}
        self.prefixes = {}  # inflection -> set(subsequences). ATTENTION: this parameter is set from the outside!
        self.nullFlex = {}

        self.lens = {}

        self.agglutinative = agglutinative

    def add_paradigm(self, paradigmObj):

        inflexions = paradigmObj.flex

        flexTypes = set()

        self.lens[paradigmObj.name] = len(inflexions)

        for flex in inflexions:

            flex.init_search_data()
            flexText = flex.comparison_pattern()
            flexTypes.add(flexText)
            grammar = flex.gramm

            if flexText != self.marker:  # this case should be processed separately;
                # otherwise, all the words from a word list are added to the dic
                self.add(flexText)
            else:
                if paradigmObj.name not in self.nullFlex:
                    self.nullFlex[paradigmObj.name] = []
                self.nullFlex[paradigmObj.name].append(grammar)

            if flexText not in self.paradigms:  # flex -> {paradigm names -> [grammars]}
                self.paradigms[flexText] = {}
            if paradigmObj.name not in self.paradigms[flexText]:
                self.paradigms[flexText][paradigmObj.name] = []
            self.paradigms[flexText][paradigmObj.name].append(grammar)

        self.pMetrics[paradigmObj.name] = len(flexTypes)

    def parse(self, token):
        """
        For a token given, get a list of options (inflection, stem) possible.

        """
        options = FlexAutomaton.parse(self, token)
        if not self.agglutinative:
            return options
        toRemove = set()
        for flex, stem in options:
            flexPrefixes = self.prefixes.get(flex, None)
            if flexPrefixes:
                toRemove |= flexPrefixes
        options = [(flex, stem) for flex, stem in options if flex not in toRemove]
        return options


class DictionaryCollector(object):
    def __init__(self, paradigmPath, conversionPath=u'', relevantParadigms=None, agglutinative=True):
        """
        :param paradigmPath: a path to a file containing paradigm description in the UniParser format.
        :param conversionPath: a path to a file describing stem conversions in the UniParser format.
        :param relevantParadigms: a lins of paradigm names to process (according to the grammar used.
        :param agglutinative: a boolean saying whether a language is agglutinative.
        :return: -

        """
        self.dics = {}

        self.conversions = {}
        self.conversionLinks = {}

        self.prefixes = {}  # { flex -> set(its prefixes) }

        g = Grammar()

        currTime = time.clock()
        if conversionPath:
            convNumber = g.load_stem_conversions(conversionPath)
        else:
            convNumber = 0
        logging.info('%d conversions loaded. Time consumed: %.4f seconds.', convNumber, time.clock() - currTime)

        currTime = time.clock()
        paradigmNumber = g.load_paradigms(paradigmPath, relevantParadigms)
        if paradigmNumber:
            logging.info('%d paradigms loaded. Time consumed: %.4f seconds.', paradigmNumber, time.clock() - currTime)
        else:
            logging.critical('%d paradigms loaded.', paradigmNumber)
            raise IOError()
        g.compile_all()

        self.automaton = _ParadigmFeaturedAutomaton(agglutinative)

        pFinder = _PrefixFinder()

        for p in g.paradigms.values():
            if not relevantParadigms or p.name in relevantParadigms:
                self.automaton.add_paradigm(p)
                pFinder.add_paradigm(p)
                if conversionPath:
                    self.conversionLinks[p.name] = p.conversion_links

        self._list_prefixes(g, pFinder)
        self.automaton.prefixes = self.prefixes
        self.conversions = g.stemConversions
        logging.info("All paradigms processed.")

    def _list_prefixes(self, grammar, prefixAutomaton):
        for p in grammar.paradigms.values():
            self._list_paradigm_prefixes(p, prefixAutomaton)

    def _list_paradigm_prefixes(self, paradigmObject, prefixAutomaton):
        """
        For each pair (flex, paradigm) add a set of subsequences of a flex in the same paradigm.

        """
        for flex in paradigmObject.flex:
            flex = flex.comparison_pattern()
            prefixes = prefixAutomaton.check(flex)
            if flex not in self.prefixes:
                self.prefixes[flex] = set()
            self.prefixes[flex] |= prefixes


    def _get_parse_options(self, wordform):
        """
        Get all the word form's parsing options with the use of the inner automaton.
        :param wordform: unicode string;
        :return: all the combinations (flex, stem) possible.

        """
        return self.automaton.parse(wordform)

    def _put_wordform_to_storage(self, whereToAddParadigms, whereToAddStems, paradigmName, stem, flex, freq):
        if paradigmName not in whereToAddParadigms:
            whereToAddParadigms[paradigmName] = set()
        whereToAddParadigms[paradigmName].add(stem)
        if stem not in whereToAddStems:
            whereToAddStems[stem] = {}
        whereToAddStems[stem][flex] = freq

    def _process_word_form(self, whereToAddParadigms, whereToAddStems, wordform, freq):
        """
        Get all the option for a word form and put them into a dic.
        :param whereToAddParadigms: {paradigm -> [stems]}
        :param whereToAddStems: {stem -> {flex -> freq}}
        :param wordform: a token to parse;
        :param freq: token's frequency in a corpora.
        :return: -

        """
        options = self._get_parse_options(wordform)
        for flex, stem in options:
            if stem != self.automaton.marker:
                paradigms = self.automaton.paradigms.get(flex, {})
                for paradigmName in paradigms:  # in: {paradigm_name -> [grammar for flex]}
                    self._put_wordform_to_storage(whereToAddParadigms, whereToAddStems, paradigmName, stem, flex, freq)

    def _add_null_flex_word_forms(self, whereToAddParadigms, whereToAddStems, fd):
        for word, freq in fd.items():
            stems = [template % word for template in (u'%s.', u'.%s', u'.%s.')]
            for paradigm in self.automaton.nullFlex:
                for stemOption in stems:
                    if paradigm in whereToAddParadigms and stemOption in whereToAddParadigms[paradigm]:
                        self._put_wordform_to_storage(whereToAddParadigms, whereToAddStems, paradigm, stemOption,
                                                      self.automaton.marker, freq)

    def _flatten_stem_list(self, stemList):
        newStemsetList = []
        for listOfListsOfStems in stemList:
            newListOfStems = set()
            for listOfStems in listOfListsOfStems:
                for stem in listOfStems:
                    newListOfStems.add(stem)
            newListOfStems = [s for s in newListOfStems if s in self.frequencies]
            newStemsetList.append(tuple(newListOfStems))
        return newStemsetList

    def _revise_paradigm(self, paradigmLex, paradigmName):
        """
        Given a list of paradigm's stems and a paradigm's name,
        compile a list of different stems per lexeme.

        """
        convLink = self.conversionLinks.get(paradigmName, None)
        if convLink:
            stems = [[[stem]] for stem in paradigmLex]  # conversion base
            for conv in convLink:
                convObj = self.conversions[conv]
                for s in stems:
                    convObj.convert(s)
            return set(self._flatten_stem_list(stems))
        # just put each lexeme into an array
        return set([(stem,) for stem in paradigmLex])

    def _merge_lexemes(self, dataCollected):
        """
        Convert a dic dataCollected ({paradigm_name -> stem list}) to {paradigm name -> list of tuples of lexeme's stems}.

        """
        for paradigmName in dataCollected:
            currentParadigm = dataCollected[paradigmName]
            refactoredParadigm = self._revise_paradigm(currentParadigm, paradigmName)
            dataCollected[paradigmName] = refactoredParadigm

    # делает из того, что лежит в памяти, нормальную лексему с указанием грамматики.
    def _add_missing_data(self, lexeme, pName):
        # I also can do something like this!
        return {stem: {flex: {"freq": self.frequencies[stem][flex],
                              "gr": self.automaton.paradigms[flex][pName]} for flex in self.frequencies[stem]
                       if pName in self.automaton.paradigms[flex]} for stem in lexeme}

    def get_extended_paradigm(self, pName):
        """
        Convert paradigm's data to the extended format acceptable for the feature extractor.
        :param pName: a name of a paradigm should be converted.
        :return: a list representing the converted paradigm.
        """
        if pName not in self.dics:
            raise ValueError("A paradigm you address is not found.")
        pDataNew = []
        for lexeme in self.dics[pName]:
            lexeme = {"lex": self._add_missing_data(lexeme, pName), "paradigm": pName}
            pDataNew.append(lexeme)
        return pDataNew

    def get_paradigm(self, pName):
        """
        Convert paradigm's data to the format used in output jsons.
        :param pName: a name of a paradigm should be converted.
        :return: a list representing the converted paradigm.

        """
        return [i["lex"] for i in self.get_extended_paradigm(pName)]

    def __getitem__(self, pName):
        """
        :param pName: a name of a paradigm to get.
        :return: a list of lexemes.
        """
        return self.get_paradigm(pName)


    def __iter__(self):
        """
        Iterate over paradigm names in the storage.

        """
        for paradigmName in self.dics:
            yield paradigmName

    def __contains__(self, item):
        """
        Check whether a paradigm in the dictionary.
        :param item: paradigm name;
        :return: a boolean.

        """
        return item in self.dics

    def process_freq_dist(self, fd):
        """
        Parse all the words in a frequency distribution and put them to a dict
        (available later through get_paradigm_to_ml() and get_paradigm() methods).
        :param fd: dict object { word form: frequency }.
        :return: -

        """
        pToStemDataCollected = {}  # {paradigm -> [stems]}
        stemToFlexDataCollected = {}  # {stem -> {flex -> freq}}
        for wf, freq in fd.items():
            self._process_word_form(pToStemDataCollected, stemToFlexDataCollected, wf, freq)
        self._add_null_flex_word_forms(pToStemDataCollected, stemToFlexDataCollected, fd)
        self.frequencies = stemToFlexDataCollected
        self._merge_lexemes(pToStemDataCollected)
        self.dics = pToStemDataCollected

        c = 0
        for paradigm in self.dics:
            c += len(self.dics[paradigm])
        logging.info(u"There are %d lexemes in the draft.", c)

    def process_word_list(self, wordList):
        """
        Parse all the words in a list and put them to a dict
        (available later through get_paradigm_to_ml() and get_paradigm() methods).
        :param wordList: a list of corpus' tokens.
        :return: -

        """
        fd = Counter(wordList)
        self.process_freq_dist(fd)

    def export_paradigm_lengths(self, path):
        """
        Save the data about the lengths of paradigms processed (in flex) to a file.
        :param path: a path to a file to export the data.
        :return: -

        """
        DataExportManager.export_to_json(self.automaton.lens, path)


class DataExportManager(object):
    @staticmethod
    def _form_filename(extension, paradigm, dir):
        paradigm = paradigm.replace(u'/', u'--')
        return os.path.join(dir, u'%s.%s' % (paradigm, extension))

    @classmethod
    def export_to_json(cls, dic, path):
        """
        Export a dictionary to a json file specified.
        :param dic: a dict to save to a file;
        :param path: a path to a file.
        :return: -

        """
        text = json.dumps(dic, ensure_ascii=False, sort_keys=True, indent=1)
        with codecs.open(path, 'w', 'utf-8') as f:
            f.write(text)

    @classmethod
    def export_to_jsons(cls, storage, dir):
        """
        Export a dictionary collected to json files in a directory specified.
        All the files are going to be rewritten if they exist!
        A name of a file is a name of a paradigm.
        :param storage: a dict or a DictionaryCollector object;
        :param dir: a path to a directory. If incorrect, IOError raised.
        :return: -

        """
        if not os.path.isdir(dir):
            raise IOError()
        if not (isinstance(storage, dict) or isinstance(storage, DictionaryCollector)):
            raise ValueError()

        for paradigm in storage:
            path = cls._form_filename(u'json', paradigm, dir)
            cls.export_to_json(storage[paradigm], path)

    @classmethod
    def _convert_stem_data(cls, stemData):
        descriptions = []
        for flex, flexData in stemData.items():
            flexLine = u'%s\t%s' % (flex, flexData["freq"])
            descriptions.append(flexLine)
            for i in flexData["gr"]:
                descriptions.append(u'\t%s' % i)
        descriptions = [u'\t' + i for i in descriptions]
        return u'\n'.join(descriptions)

    @classmethod
    def _convert_lexeme(cls, lexeme):
        return u'\n'.join([stem + u'\n' + cls._convert_stem_data(lexeme[stem]) for stem in lexeme])

    @classmethod
    def _export(cls, pContents, path):
        lexemes = [cls._convert_lexeme(lexeme) for lexeme in pContents]
        text = u'\n\n'.join(lexemes)
        with codecs.open(path, 'w', 'utf-8') as f:
            f.write(text)

    @classmethod
    def export_to_txt(cls, storage, dir):
        """
        Export a dictionary collected to plain text files in a directory specified.
        The frequencies will be saved, too. All the files are going to be rewritten if they exist!
        :param storage: a dict or a DictionaryCollector object;
        :param dir: a path to a directory.
        :return: -

        """
        if not os.path.isdir(dir):
            raise IOError()
        if not (isinstance(storage, dict) or isinstance(storage, DictionaryCollector)):
            raise ValueError()

        for paradigm in storage:
            path = cls._form_filename(u'txt', paradigm, dir)
            cls._export(storage[paradigm], path)


class DraftCleaner(object):
    """ The class contains methods to clear the data from trash.
    Requires an object initialization to use methods.

    """

    def __init__(self, pathToTraining, pathToParadigmLengths, pathToCategories):
        self.transformer = DataTransformer(pathToTraining, pathToParadigmLengths, pathToCategories)

    @staticmethod
    def threshold_cleaning(dic_collector_obj):
        """ Clear the data in a grdicmaker.DictionaryCollector object. If a storage is empty, return None.
        The method is static and does not require instance initialization.

        """
        if not dic_collector_obj.dics:
            return None
        new_dic = {}
        for paradigm_name in dic_collector_obj.dics:
            if paradigm_name not in new_dic:
                new_dic[paradigm_name] = []
            for lex in dic_collector_obj.get_paradigm(paradigm_name):
                if Thresholds.evaluate(lex):
                    new_dic[paradigm_name].append(lex)
        return new_dic

    def _train_svm(self, cVal, kernelVal, exclusion, ablation_features=()):
        headlines, data, targets = self.transformer.get_training_data_matrix(normalize=True,
                                                                             ablation_features=ablation_features,
                                                                             toExclude=set(exclusion))
        classifier = svm.SVC(C=cVal, kernel=kernelVal)
        classifier.fit(data, targets)
        return classifier

    def _train_linear(self, exclusion, ablation_features=()):
        headlines, data, targets = self.transformer.get_training_data_matrix(normalize=False,
                                                                             ablation_features=ablation_features,
                                                                             toExclude=set(exclusion))
        classifier = linear_model.LinearRegression()
        classifier.fit(data.toarray(), targets)
        return classifier

    def _train_perceptron(self, exclusion, ablation_features=()):
        headlines, data, targets = self.transformer.get_training_data_matrix(normalize=False,
                                                                             ablation_features=ablation_features,
                                                                             toExclude=set(exclusion))
        classifier = linear_model.Perceptron()
        classifier.fit(data.toarray(), targets)
        return classifier

    def _train_forest(self, exclusion, ablation_features=()):
        headlines, data, targets = self.transformer.get_training_data_matrix(normalize=True,
                                                                             ablation_features=ablation_features,
                                                                             toExclude=set(exclusion))
        classifier = RandomForestClassifier(n_estimators=10)
        classifier.fit(data.toarray(), targets)
        return classifier

    def _classifier_clearing(self, dicCollectorObj, categories, classifierTrained, normalize, ablation_features=()):
        newDic = {}
        for pName in dicCollectorObj.dics:
            # this is a little shitty piece
            # converting the data to the appropriate format as the ML module requires different data format
            dataML = dicCollectorObj.get_extended_paradigm(pName)
            headlines, featureMatrix = self.transformer.get_processing_data_matrix(dataML, dicCollectorObj, categories,
                                                                                   normalize, ablation_features)
            results = classifierTrained.predict(featureMatrix.toarray())
            clearedParadigm = [
                {
                    "lex": dataML[i]["lex"],
                    "guess": (True if int(results[i]) > 0 else False)
                } for i in xrange(len(results))]
            newDic[pName] = clearedParadigm
        return newDic

    def svm_cleaning(self, dicCollectorObj, categories, exclusion=(), ablation=()):
        """ Apply SVM classification to the data of a dictionary collected.
        Args:
        - dicCollectorObj: grdicmaker.DictionaryCollector object, the inner dic should not be empty;
        - categories: a string, a path to a file describing category options;
        Optional args (use carefully, as they were designed to do experiments):
        - exclusion: a list of paradigm names should be excluded from the training data;
        - ablation: list of strings, names of features to ablate (see a list of possible parameters in the description of DataTransformer class).
        Return: a json-like dictionary structure.

        """
        if not dicCollectorObj.dics:
            return None
        trainedSVM = self._train_svm(1, 'rbf', exclusion, ablation)  # classifier params are static but it may change
        return self._classifier_clearing(dicCollectorObj, categories, trainedSVM, normalize=True,
                                         ablation_features=ablation)

    def forest_clearing(self, dicCollectorObj, categories, exclusion=(), ablation=()):
        """Apply Random Forest classification to the data of a dictionary collected.
        Args and return: see DraftCleaner.svm_cleaning().

        """
        if not dicCollectorObj.dics:
            return None
        forest = self._train_forest(exclusion, ablation)
        return self._classifier_clearing(dicCollectorObj, categories, forest, normalize=True,
                                         ablation_features=ablation)

    def linear_clearing(self, dicCollectorObj, categories, exclusion=(), ablation=()):
        """Apply Linear Regression classification to the data of a dictionary collected.
        Args and return: see DraftCleaner.svm_cleaning().

        """
        if not dicCollectorObj.dics:
            return None
        linearModel = self._train_linear(exclusion, ablation)
        return self._classifier_clearing(dicCollectorObj, categories, linearModel, normalize=False,
                                         ablation_features=ablation)

    def perceptron_clearing(self, dicCollectorObj, categories, exclusion=(), ablation=()):
        """Apply perceptron classification to the data of a dictionary collected.
        Args and return: see DraftCleaner.svm_cleaning().

        """
        if not dicCollectorObj.dics:
            return None
        perceptron = self._train_perceptron(exclusion, ablation)
        return self._classifier_clearing(dicCollectorObj, categories, perceptron, normalize=False,
                                         ablation_features=ablation)


class DataTransformer(object):
    """ A class to transform the data to a feature matrix. Feature inner names (used to ablate a feature) are:
    1. 'entropy': an entropy of a frequency distribution of a lexeme's inflections;
    2. 'freq_proportion': a fraction (number of unique lexeme's word forms) / (number of occurences in a corpus);
    3. 'average_category_entropy': an average value of grammatical categories' length-normalized entropies;
    4. 'min_category_entropy': a munimum value of grammatical categories' length-normalized entropies;
    5. 'found_flex_part': a part of a paradigm's inflections occurred in a lexeme;
    6. 'found_gramm_part': a part of paradigm's grammatical forms supposed which occurred in a lexeme;
    7. 'entropy_to_paradigm_length':  an entropy of a frequency distribution of a lexeme's inflections divided by a paradigm's length;
    8. 'number_of_one_value_categories': a number of categories which have one value only;
    9. 'category_entropy_variance': a variance of grammatical categories' length-normalized entropies.

    """

    def __init__(self, pathToTraining, pathToParadigmLengths, pathToCategories):
        """ Args:
        - pathToTraining: a path to the main training data;
        - pathToParadigmLengths: path to paradigm length description;
        - pathToCategories: path to category value description.

        """

        # these are the variables used to get TRAINING data.
        self.MLDataPath = pathToTraining
        self.paradigmLengthsPath = pathToParadigmLengths
        self.categoryPath = pathToCategories
        # and this is a full feature set
        self.features = {
            'entropy': FeatureExtractor.entropy,
            'freq_proportion': FeatureExtractor.proportion_flex_token,
            'average_category_entropy': self._average_category_entropy,
            'min_category_entropy': self._min_category_entropy,
            'found_flex_part': self._part_of_found_flex,
            'found_gramm_part': self._part_of_found_gramm,
            'entropy_to_paradigm_length': self._entropy_to_paradigm_length,
            'number_of_one_value_categories': self._number_of_one_value_categories,
            'category_entropy_variance': self._category_entropy_variance
        }
        # these are temporary data variables.
        self.pLengths = None
        self.categoryDescription = None

    def _read_category_val_alternations(self, caregoryFileName):
        with codecs.open(caregoryFileName, 'r', 'utf-8-sig') as f:
            catDistr = json.loads(f.read())
            self.categoryDescription = {}
            for element in catDistr:
                for paradigmName in element["paradigms"]:
                    for category in element["categories"]:
                        category = set(category["values"])
                        # assert len(category) > 1
                        if paradigmName not in self.categoryDescription:
                            self.categoryDescription[paradigmName] = []
                        self.categoryDescription[paradigmName].append(category)

    def _read_paradigm_lengths(self):
        with codecs.open(self.paradigmLengthsPath, 'r', 'utf-8-sig') as f:
            self.pLengths = json.loads(f.read())

    def _copy_paradigm_lengths(self, dataCollectorObject):
        self.pLengths = dataCollectorObject.automaton.pMetrics

    # ------ these are wrappers for the funcs requiring more arguments... -------
    def _average_category_entropy(self, lexeme):
        return FeatureExtractor.average_category_entropy(lexeme, self.categoryDescription)

    def _min_category_entropy(self, lexeme):
        return FeatureExtractor.min_category_entropy(lexeme, self.categoryDescription)

    def _part_of_found_flex(self, lexeme):
        return FeatureExtractor.part_of_found_flex(lexeme, self.pLengths)

    def _part_of_found_gramm(self, lexeme):
        return FeatureExtractor.part_of_found_grammars(lexeme, self.pLengths)

    def _entropy_to_paradigm_length(self, lexeme):
        return FeatureExtractor.entropy_to_paradigm_length(lexeme, self.pLengths)

    def _number_of_one_value_categories(self, lexeme):
        return FeatureExtractor.number_of_one_value_categories(lexeme, self.categoryDescription)

    def _category_entropy_variance(self, lexeme):
        return FeatureExtractor.category_entropy_variance(lexeme, self.categoryDescription)

    # --------------------------------------------------------------------------

    def _check_if_ablation_appropriate(self, ablation):
        for feature in ablation:
            if feature not in self.features:
                raise KeyError('Invalid feature set required.')

    def _convert_lexeme_to_feature_dic(self, lexeme, ablation_features):
        dic = {funcName: self.features[funcName](lexeme) for funcName in self.features if
               funcName not in ablation_features}
        return dic

    # def _dic_list_to_matrix(self, processedData, normalize):
    #     vectorizer = DictVectorizer()
    #     if normalize:
    #         return preprocessing.normalize(vectorizer.fit_transform(processedData), norm='l2')
    #     return vectorizer.fit_transform(processedData)

    def _dic_list_to_matrix(self, processedData, normalize):
        vectorizer = DictVectorizer()
        if normalize:
            res = preprocessing.normalize(vectorizer.fit_transform(processedData), norm='l2')
        else:
            res = vectorizer.fit_transform(processedData)
        return vectorizer.get_feature_names(), res

    def get_training_data_matrix(self, normalize, ablation_features=(), toExclude=()):
        """ Process the training data.
        Args:
        — normalize: a boolean flag if the data should be normalized in a feature matrix;
        — toExclude: a list/tuple of paradigms that cannot be used in the training data (for cross-validation);
        — ablate: a list of features to exclude during the ablation study.
        Return:
        - headlines;
        - a training sparse scipy matrix;
        - a list of targets.


        """
        assert isinstance(normalize, bool)
        self._check_if_ablation_appropriate(ablation_features)

        # additional data initialization:
        self._read_category_val_alternations(self.categoryPath)  # self.categoryDescription
        self._read_paradigm_lengths()  # self.pLengths
        # and this is a table maker itself
        setParadigms = set()
        with codecs.open(self.MLDataPath, 'r', 'utf-8-sig') as f:
            data = json.loads(f.read())
            processedData = []
            targets = []
            for lexeme in data:
                if lexeme["paradigm"] in toExclude:
                    continue
                else: setParadigms.add(lexeme["paradigm"])

                lexemeFeatureDic = self._convert_lexeme_to_feature_dic(lexeme, ablation_features)
                processedData.append(lexemeFeatureDic)

                sampleEval = FeatureExtractor.is_positive_example(lexeme)
                targets.append(sampleEval)

            headlines, matrix = self._dic_list_to_matrix(processedData, normalize)
            if setParadigms:
                logging.info("Training set paradigms: %s", u" ".join(list(setParadigms)))
            else:
                logging.critical("Training set is empty.")

            return headlines, matrix, targets

    def get_processing_data_matrix(self, data, dataCollectorObject, categoryDescription, normalize, ablation_features):
        """ Args:
        - data: a list of lexemes in ml compatible json;
        - dataCollectorObject: an object containing some basic info about the data collection;
        - categoryDescription;
        - normalize: if the data should be normalized;
        - ablation_features: a list of features to eliminate to conduct an ablation study.
        Return:
        - a sparse scipy matrix.

        """
        self._read_category_val_alternations(categoryDescription)
        self._copy_paradigm_lengths(dataCollectorObject)
        processedData = [self._convert_lexeme_to_feature_dic(lexeme, ablation_features) for lexeme in data]
        return self._dic_list_to_matrix(processedData, normalize)


class DatasetAnnotator(object):
    """ This class contains tools to evaluate the data got from the dictionary collector.
    The data should be saved to files which names are in accordance with the names of paradigms in a dictionary
    and put into one directory. An initialization of an instance is required to use both features (annotation or evaluation).

    """

    def __init__(self):
        self.annotatedJson = []
        self.approved, self.notApproved = 0, 0

    @staticmethod
    def _sum_lex_freq(lexeme):
        summ = 0
        for stem in lexeme:
            flexDic = lexeme[stem]
            for flex in flexDic:
                summ += flexDic[flex]["freq"]
        return summ

    def _sample_list(self, lst, numOfSamples):
        if numOfSamples is None:
            return lst
        if len(lst) < numOfSamples:
                raise AssertionError
        return random.sample(lst, numOfSamples)

    def _compile_evaluated_dic(self, pathToDir, dic, threshold, pSampleNum, nSampleNum, weight_func):

        filenameSample = u'%s.json'

        approvedSamples, notApprovedSamples = [], []

        for paradigmName in dic.list_paradigms():
            fn = filenameSample % paradigmName
            path = os.path.join(pathToDir, fn)
            if not os.path.exists(path):
                continue
            with codecs.open(path, 'r', 'utf-8') as f:
                parsedData = json.loads(f.read())
                for lexeme in parsedData:
                    lexFreq = self._sum_lex_freq(lexeme)
                    if lexFreq < threshold:
                        continue

                    isAppropriate = dic.look_up(lexeme, paradigmName)
                    updLex = {"lex": lexeme, "eval": (isAppropriate or isAppropriate), "paradigm": paradigmName}
                    if isAppropriate:
                        approvedSamples.append(updLex)
                    else:
                        notApprovedSamples.append(updLex)

        positiveSamplesToUse = self._sample_list(approvedSamples, pSampleNum)
        negativeSamplesToUse = self._sample_list(notApprovedSamples, nSampleNum)
        # compile final JSON
        self.annotatedJson = positiveSamplesToUse + negativeSamplesToUse
        random.shuffle(self.annotatedJson)
        # Finally, update statistics
        positiveWeight, negativeWeight = 0, 0
        if positiveSamplesToUse:
            positiveWeight = sum([weight_func(lex["lex"]) for lex in positiveSamplesToUse])
        if negativeSamplesToUse:
            negativeWeight = sum([weight_func(lex["lex"]) for lex in negativeSamplesToUse])
        self.approved = positiveWeight
        self.notApproved = negativeWeight

    def _annotate_paradigm_data(self, data, paradigm, lookup):
        """
        Add a label "eval" to each lexeme in a data array.
        :param data: an output of a classifier (a json: {"lex": ..., "guess": bool})
        :param paradigm: a name of the paradigm;
        :param lookup: a dictionary;
        :return: -

        """
        for lexeme_box in data:
            lex = lexeme_box["lex"]
            is_appropriate = lookup.look_up(lex, paradigm)
            lexeme_box["eval"] = is_appropriate

    def annotate(self, dictionary, with_lookup):
        """
        Complete an input dictionary with labels showing whether a lexeme is correct according to a dictionary used.
        :param dictionary: {"paradigm-name": {"lex": ..., "guess": ...}}
        :param with_lookup: a dictionary having a look_up() method.
        :return: -

        """
        for paradigm in dictionary:
            self._annotate_paradigm_data(dictionary[paradigm], paradigm, with_lookup)

    def evaluate(self, pathToInput, lookUp, weighted, threshold=STANDARD_THRESHOLD):
        """ Evaluate the data and show the weights of approved and disproved data.
        Args:
        - pathToInput: path to an input directory containing json data to evaluate;
        - lookUp: an instance having a look_up() method to check if a lexeme is a valid one.
        Instances of the class should not require any args while initializing;
        - weighted: a boolean parameter showing whether a number of (in)valid lexemes should be count
        or a number of their occurrences in the corpus.
        - threshold: a minimum frequency a sample should have to be added to the output. The default is STANDARD_THRESHOLD.

        """
        if not weighted:
            self._compile_evaluated_dic(pathToInput, lookUp, threshold, None, None, weight_func=lambda a: 1)
        else:
            self._compile_evaluated_dic(pathToInput, lookUp, threshold, None, None, weight_func=DatasetAnnotator._sum_lex_freq)
        return self.approved, self.notApproved

    def _annotate(self, pathToInput, pathToOutput, lookUp, positiveSampleNum=None, negativeSampleNum=None, threshold=STANDARD_THRESHOLD):
        """
        Having evaluated the data, save the necessary number of samples to a file.
        Args:
        - pathToInput: path to an input directory containing json data to evaluate;
        - pathToOutput: path to a file where a data set compiled should be written;
        - lookUp: an instance having a look_up() method to check if a lexeme is a valid one.
        Instances of the class should not require any args while initializing;
        - positiveSampleNum: a number of positive samples in the output.
        The default is None; this means that all the positive samples will be included,
        - negativeSampleNum: a number of negative samples in the output. Default is None, like for positive ones;
        - threshold: a minimum frequency a sample should have to be added to the output. The default is STANDARD_THRESHOLD.

        """
        self._compile_evaluated_dic(pathToInput, lookUp, threshold, positiveSampleNum, negativeSampleNum, weight_func=lambda a: 1)
        text = json.dumps(self.annotatedJson, encoding="utf-8", ensure_ascii=False, sort_keys=True, indent=1)
        with codecs.open(pathToOutput, 'w', 'utf-8') as f:
            f.write(text)

    def create_testset(self, wfs, paradigms, conv, agglutinative, pathToOutput, lookUp,
                       positiveSampleNum=None, negativeSampleNum=None, threshold=STANDARD_THRESHOLD):
        """
        Generate an annotated data set and save it to a directory.
        :param wfs: a list of word forms;
        :param paradigms: a path to paradigm descriptions;
        :param conv: a path to conversion description;
        :param agglutinative: a boolean;
        :param pathToOutput: a path to save the results to;
        :param lookUp: a dictionary having a look_up() method;
        :param positiveSampleNum: a number of positive samples required (by default, all the samples are saved);
        :param negativeSampleNum: a number of negative samples required (by default, all the samples are saved);
        :param threshold: a minimum frequency a sample should have to be added to the output. The default is STANDARD_THRESHOLD.
        :return:

        """
        dc = DictionaryCollector(paradigms, conv, agglutinative=agglutinative)
        dc.process_word_list(wfs)
        tdir = tempfile.mkdtemp(dir=os.getcwd())
        try:
            DataExportManager.export_to_jsons(dc, tdir)
            self._annotate(tdir, pathToOutput, lookUp, positiveSampleNum, negativeSampleNum, threshold)
        finally:
            shutil.rmtree(tdir)


if __name__ == '__main__':
    import random
    import data_accessors

    testWordforms, paradigms, conv = data_accessors.LangTestData.kazakh()
    fd = Counter(testWordforms)

    # dc = DictionaryCollector(paradigms, conv, agglutinative=True)
    #
    # words_allowed = list(set(testWordforms))
    # for share in [0.25, 0.5, 0.75, 1]:
    #     random.shuffle(words_allowed)
    #     word_number = int(len(words_allowed) * share)
    #     print "A corpus size: %d" % word_number
    #     current_fd = {i: fd[i] for i in words_allowed[:word_number]}
    #     curr_time = time.clock()
    #     dc.process_freq_dist(current_fd)
    #     print "Time: %f" % (time.clock() - curr_time)

    dc = DictionaryCollector(paradigms, conv, relevantParadigms=["N-soft"], agglutinative=True)
    curr_time = time.clock()
    dc.process_freq_dist(fd)
    print "Time for N-soft: %f" % (time.clock() - curr_time)
    DataExportManager.export_to_json(dc.get_extended_paradigm("N-soft"), "trashbin/ns.json")
    # DataExportManager.export_to_txt(dc, "trashbin")

    # testWordforms, paradigms, conv = test_data_readers.LangTestData.kazakh_nouns()
    # dc = DictionaryCollector(paradigms, conv, agglutinative=True)
    # curr_time = time.clock()
    # dc.process_freq_dist(fd)
    # print "Time for N: %f" % (time.clock() - curr_time)
    #
    # testWordforms, paradigms, conv = test_data_readers.LangTestData.kazakh_verbs()
    # dc = DictionaryCollector(paradigms, conv, agglutinative=True)
    # curr_time = time.clock()
    # dc.process_freq_dist(fd)
    # print "Time for V: %f" % (time.clock() - curr_time)


