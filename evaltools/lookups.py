# coding: utf-8

"""
A module containing a base dictionary access class, and a series of standard dictionary classes.

"""

import codecs
import os
import re


class LookupDic(object):
    """
    A base class of a dictionary.

    """

    STEM_REGEX = re.compile(u'stem: ([^\s]+)\s', flags = re.I|re.U)
    LEX_REGEX = re.compile(u'lex: ([^\s]+)\s', flags = re.I|re.U)
    PARADIGM_REGEX = re.compile(u'paradigm: ([^\s]+)\s', flags = re.I|re.U)
    POS_REGEX = re.compile(u'gramm: ([^\s]+)\s', flags = re.I|re.U)

    def __init__(self, path):
        """
        :param path: a path to a text version of a dictionary. The standard format is YAML
        (see examples in a 'dictionaries' dir).
        :return: -

        """
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
        """
        Check if a lexeme belongs to a paradigm.
        :param lexeme: a dictionary representing a lexeme.
        :param pName: a name of the paradigm.
        :return: a boolean.

        """
        stemIterable = lexeme.keys()
        appropriates = self.maindic[pName]
        return self._approve_main(stemIterable, appropriates)

    def list_paradigms(self):
        """
        List dictionary's paradigms.
        :return: a list of paradigms included in the dictionary.

        """
        return self.maindic.keys()


class LookupDicKZ(LookupDic):
    """
    A class giving access to a dictionary of Kazakh.

    """

    def __init__(self):
        yamlDicPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), u'dictionaries/kz/kz_lexemes.txt')
        LookupDic.__init__(self, yamlDicPath)
        self.lexListPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), u'dictionaries/kz/AdditionalLexemes.txt')
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
    """
    A class giving access to a dictionary of Katharevousa.

    """
    def __init__(self):
        yamlDicPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), u'dictionaries/katharevousa/adj_os1.txt')
        LookupDic.__init__(self, yamlDicPath)


class LookupDicAlbanian(LookupDic):
    """
    A class giving access to a dictionary of Albanian.

    """
    def __init__(self):
        yamlDicPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), u'dictionaries/albanian/stems-all-cleaned.txt')
        LookupDic.__init__(self, yamlDicPath)


class LookupDicUdmurt(LookupDic):
    """
    A class giving access to a dictionary of Udmurt.

    """
    def __init__(self):
        yamlDicPaths = os.path.join(os.path.dirname(os.path.abspath(__file__)), u'dictionaries/udmurt')
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