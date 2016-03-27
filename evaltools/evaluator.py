# coding: utf-8

"""
A module containing instruments for data evaluation/ annotation.

"""

import codecs
import json
import os
import random

STANDARD_THRESHOLD = 0


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
            self._compile_evaluated_dic(pathToInput, lookUp, threshold, None, None, weight_func=Evals._sum_lex_freq)
        return self.approved, self.notApproved

    def annotate(self, pathToInput, pathToOutput, lookUp, positiveSampleNum=None, negativeSampleNum=None, threshold=STANDARD_THRESHOLD):
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
