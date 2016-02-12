# coding: utf-8

"""
A module containing instruments for data evaluation/ annotation.

"""

import codecs
import json
import random
import re
import os


STANDARD_THRESHOLD = 0


class LookupDic(object):

    STEM_REGEX = re.compile(u'stem: ([^\s]+)\s', flags = re.I|re.U)
    LEX_REGEX = re.compile(u'lex: ([^\s]+)\s', flags = re.I|re.U)
    PARADIGM_REGEX = re.compile(u'paradigm: ([^\s]+)\s', flags = re.I|re.U)
    POS_REGEX = re.compile(u'gramm: ([^\s]+)\s', flags = re.I|re.U)

    def __init__(self, path):
        self._set_renamer()
        self.yamlDicPath = path
        self.maindic = self._compile_dic_from_yaml()

    def _set_renamer(self):
        self.renamer = {}

    def _read_yaml_dic(self):
        """ Read a dictionary in the YAML format.
        Yield tuples:
        (lexeme (unicode), paradigm name (unicode), pos (unicode), lexeme's stems (tuple of unicode strings)).

        """
        with codecs.open(self.yamlDicPath, 'r', 'utf-8') as f:
            text = f.read()
            rawLexemes = re.split(u'\s+-lexeme', text)
            for l in rawLexemes:
                lex = self.LEX_REGEX.search(l)
                paradigmTypes = self.PARADIGM_REGEX.findall(l)  # todo test this change
                gr = self.POS_REGEX.search(l)
                if not lex or not paradigmTypes or not gr:
                    continue
                lex, gr = lex.group(1), gr.group(1)
                for pType in paradigmTypes:
                    lexStems = self.STEM_REGEX.findall(l)
                    lexStems = [tuple([j for j in i.replace(u'/', u'|').split(u'|') if j]) for i in lexStems]
                    pType = self.renamer.get(pType, pType)
                    yield lex, pType, gr, lexStems

    @staticmethod
    def _reorganize_stem_lst(stemIterable):
        """ Convert a stem list arranged hierarchically to a set of all the stems.
         Args:
         - a list of iterables of stems.
         Return: set()

        """
        flatStemset = set()
        for stNum in stemIterable:
            for stOption in stNum:
                flatStemset.add(stOption)
        return flatStemset

    def _compile_dic_from_yaml(self):
        pLexemes = {}
        # paradigm name -> {stem -> all the stems may be found in this paradigm}
        for lex, ptype, gr, lexStems in self._read_yaml_dic():
            # if lex in lexDic and gr in lexDic[lex]:
                if ptype not in pLexemes:
                    pLexemes[ptype] = {}
                allStems = self._reorganize_stem_lst(lexStems)
                for i in allStems:
                    if i not in pLexemes[ptype]:
                        pLexemes[ptype][i] = set()
                    pLexemes[ptype][i] |= allStems
        return pLexemes

    @staticmethod
    def _approve_main(stemIterable, appropriates):
        """ this is a main func approving or rejecting a word. Args:
        - list of stems representing a lexeme;
        - list of appropriate stems.

        """
        if not stemIterable:
            return False
        if stemIterable[0] in appropriates:
            for i in stemIterable:
                if i not in appropriates[stemIterable[0]]:
                    return False
            return True
        else:
            return False

    def look_up(self, lexeme, pName):
        """ Check if a lexeme belongs to a paradigm.
        Args:
        - lexeme;
        - paradigm name.

        """
        stemIterable = lexeme.keys()
        appropriates = self.maindic[pName]
        return self._approve_main(stemIterable, appropriates)

    def list_paradigms(self):
        """Get a list of paradigms found in a dictionary.

        """
        return self.maindic.keys()


class LookupDicKZ(LookupDic):

    def __init__(self):
        yamlDicPath = os.path.join(os.path.abspath(__file__), u'dictionaries/kz/kz_lexemes.txt')
        LookupDic.__init__(self, yamlDicPath)
        self.lexListPath = os.path.join(os.path.abspath(__file__), u'dictionaries/kz/AdditionalLexemes.txt')
        self.additionalDic = self._compile_dic_from_list()

    def _set_renamer(self):
        self.renamer = {'V-hard': 'V-mood-hard', 'V-soft': 'V-mood-soft'}

    @staticmethod
    def _kzstem(wf, pos):
        if pos == 'V':
            if wf.endswith(u'у') or wf.endswith(u'ю'):
                return wf[:-1] + u'.'
        elif pos == 'N':
            return wf + u'.'

    def _compile_dic_from_list(self):
        dic = set()
        with codecs.open(self.lexListPath, 'r', 'utf-8-sig') as f:
            for line in f:
                line = line.strip()
                try:
                    lex, paradigm, ptype = line.split()
                    stem = self._kzstem(lex, paradigm)
                    if not stem:
                        continue
                    pFull = u'%s-%s' % (paradigm, ptype)
                    tup = stem, self.renamer.get(pFull, pFull)
                    dic.add(tup)
                except ValueError:
                    pass
        return dic

    @staticmethod
    def _approve_additional(stemIterable, paradigm, appropriates):
        if not stemIterable:
            return False
        for stem in stemIterable:
            tup = stem, paradigm
            if tup in appropriates:
                return True
        return False

    def look_up(self, lexeme, pName):
        isInMain = LookupDic.look_up(self, lexeme, pName)
        isInAdditional = self._approve_additional(lexeme.keys(), pName, self.additionalDic)
        return (isInAdditional or isInMain)


class LookupDicKatharevousa(LookupDic):
    def __init__(self):
        yamlDicPath = os.path.join(os.path.abspath(__file__), u'dictionaries/katharevousa/adj_os1.txt')
        LookupDic.__init__(self, yamlDicPath)


class LookupDicAlbanian(LookupDic):
    def __init__(self):
        yamlDicPath = os.path.join(os.path.abspath(__file__), u'dictionaries/albanian/stems-all-cleaned.txt')
        LookupDic.__init__(self, yamlDicPath)


class LookupDicUdmurt(LookupDic):
    def __init__(self):
        yamlDicPaths = os.path.join(os.path.abspath(__file__), u'dictionaries/udmurt')
        LookupDic.__init__(self, yamlDicPaths)

    def _read_yaml_dic(self):
        """ Read a dictionary in the YAML format.
        Yield tuples:
        (lexeme (unicode), paradigm name (unicode), pos (unicode), lexeme's stems (tuple of unicode strings)).

        """
        assert os.path.isdir(self.yamlDicPath)
        path = self.yamlDicPath
        dictionaries = [os.path.join(path, i) for i in os.listdir(path)]
        dictionaries = filter(lambda a: (os.path.isfile(a) and a.endswith(u'.txt')), dictionaries)
        for dicPath in dictionaries:
            self.yamlDicPath = dicPath
            for i in LookupDic._read_yaml_dic(self):
                yield i
        self.yamlDicPath = path


class Evals(object):
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

    def evaluate(self, pathToInput, lookUpClass, weighted, threshold=STANDARD_THRESHOLD):
        """ Evaluate the data and show the weights of approved and disproved data.
        Args:
        - pathToInput: path to an input directory containing json data to evaluate;
        - lookUpClass: a name of a class having a look_up() method to check if a lexeme is a valid one.
        Instances of the class should not require any args while initializing;
        - weighted: a boolean parameter showing whether a number of (in)valid lexemes should be count
        or a number of their occurrences in the corpus.
        - threshold: a minimum frequency a sample should have to be added to the output. The default is STANDARD_THRESHOLD.

        """
        dic = lookUpClass()
        if not weighted:
            self._compile_evaluated_dic(pathToInput, dic, threshold, None, None, weight_func=lambda a: 1)
        else:
            self._compile_evaluated_dic(pathToInput, dic, threshold, None, None, weight_func=Evals._sum_lex_freq)
        return self.approved, self.notApproved

    def annotate(self, pathToInput, pathToOutput, lookUpClass, positiveSampleNum=None, negativeSampleNum=None, threshold=STANDARD_THRESHOLD):
        """ Having evaluated the data, save the necessary number of samples to a file.
        Args:
        - pathToInput: path to an input directory containing json data to evaluate;
        - pathToOutput: path to a file where a data set compiled should be written;
        - lookUpClass: a name of a class having a look_up() method to check if a lexeme is a valid one.
        Instances of the class should not require any args while initializing;
        - positiveSampleNum: a number of positive samples in the output.
        The default is None; this means that all the positive samples will be included,
        - negativeSampleNum: a number of negative samples in the output. Default is None, like for positive ones;
        - threshold: a minimum frequency a sample should have to be added to the output. The default is STANDARD_THRESHOLD.

        """
        dic = lookUpClass()
        self._compile_evaluated_dic(pathToInput, dic, threshold, positiveSampleNum, negativeSampleNum, weight_func=lambda a: 1)
        text = json.dumps(self.annotatedJson, encoding="utf-8", ensure_ascii=False, sort_keys=True, indent=1)
        with codecs.open(pathToOutput, 'w', 'utf-8') as f:
            f.write(text)
