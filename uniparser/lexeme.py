import wordform
from afx_table import AfxInTable
import copy

def check_compatibility(lex, sublex, flex, errorHandler=None):
    """Check if the given SubLexeme and the given Inflexion
    are compatible."""
    if flex.stemNum != None and sublex.numStem != flex.stemNum:
        return False
    for rxTest in flex.regexTests:
        if check_for_regex(lex, sublex, rxTest, errorHandler) == False:
            return False
    return True

def check_for_regex(lex, sublex, rxTest, errorHandler=None):
    """Perform the given RegexTest against the given SubLexeme."""
    if rxTest.field == u'stem' or rxTest.field == u'prev':
        if rxTest.perform(sublex.stem) == False:
            return False
    elif rxTest.field == u'paradigm':
        if errorHandler != None:
            errorHandler.RaiseError(u'Paradigm names cannot be subject to ' +\
                                    u'regex tests.')
        return False
    elif rxTest.field in Lexeme.propertyFields:
        searchField = rxTest.field
        if searchField == u'lex':
            searchField = u'lemma'
        if rxTest.perform(lex.__dict__[searchField]) == False:
            return False
    else:
        testResults = [rxTest.perform(d[1])\
                       for d in lex.otherData\
                       if d[0] == rxTest.field]
        if len(testResults) <= 0 or not all(testResults):
            return False
    return True



class SubLexeme:
    def __init__(self, numStem, stem, paradigm, gramm, gloss):
        self.numStem = numStem      # the number of the stem
        self.stem = stem
        self.paradigm = paradigm
        self.gramm = gramm
        self.gloss = gloss

    def make_stem(self, lex, flexInTable):
        """Insert the inflexion parts from the (middle) inflexion
        into the current stem and return the result as an AfxInTable object
        or None if the inflexion and the stem aren't compatible.
        If the stem starts with a dot, or ends with a dot, those are deleted.
        """
        if check_compatibility(lex, self, flexInTable.afx):
            return None
        middleStem = self.stem
        if middleStem.startswith(u'.'):
            middleStem = middleStem[1:]
        if middleStem.endswith(u'.'):
            middleStem = middleStem[:-1]
        wf, wfGlossed, gloss = wordform.join_stem_flex(middleStem,
                                                       self.gloss,
                                                       flexInTable.afx,
                                                       bStemStarted=True)
        stemInTable = AfxInTable()
        stemInTable.afx = wf
        stemInTable.grFeatures = flexInTable.grFeatures
        stemInTable.lexFeatures = set([(lex.lemma, lex.gramm, wfGlossed,
                                        gloss, self.numStem)])
        return stemInTable

class ExceptionForm:
    def __init__(self, dictDescr, errorHandler=None):
        self.form = u''
        self.gramm = u''
        self.coexist = False
        self.errorHandler = errorHandler
        try:
            gramm = dictDescr[u'value']
            if dictDescr[u'content'] != None:
                for obj in dictDescr[u'content']:
                    if obj[u'name'] == u'coexist':
                        if obj[u'value'] == u'yes':
                            self.coexist = True
                        elif obj[u'value'] == u'no':
                            self.coexist = False
                        elif self.errorHandler != None:
                            self.errorHandler.RaiseError(u'The coexist field must ' +\
                                                         u'have yes or no as its value: ',\
                                                         dictDescr)
                    elif obj[u'name'] == u'form':
                        self.form = obj[u'value']
        except KeyError:
            if self.errorHandler != None:
                self.errorHandler.RaiseError(u'Exception description error: ', dictDescr)
                return
        if len(self.form) <= 0 and self.errorHandler != None:
            self.errorHandler.RaiseError(u'No form provided in an exception description: ',\
                                         dictDescr)
    def __neq__(self, other):
        if type(other) != ExceptionForm:
            return False
        if other.form == self.form and other.gramm == self.gramm and\
           other.coexist == self.coexist:
            return True
        return False

    def __eq__(self, other):
        return not self.__neq__(other)


class Lexeme:
    obligFields = set([u'lex', u'stem', u'paradigm'])
    propertyFields = set([u'lex', u'stem', u'paradigm', u'gramm', u'gloss',\
                          u'lexref'])
    g = None # Grammar object
        
    def __init__(self, dictDescr, errorHandler=None):
        self.lemma = u''
        self.lexref = u''
        self.stem = u''
        self.paradigms = []
        self.gramm = u''
        self.gloss = u''
        self.subLexemes = []
        self.exceptions = {} # set of tags -> ExceptionForm object
        self.otherData = [] # list of tuples (name, value)
        self.key2func = {u'lex': self.add_lemma, u'lexref': self.add_lexref,\
                         u'stem': self.add_stem, u'paradigm': self.add_paradigm,\
                         u'gramm': self.add_gramm, u'gloss': self.add_gloss,\
                         u'except': self.add_except}
        self.errorHandler = errorHandler
        try:
            keys = set(obj[u'name'] for obj in dictDescr[u'content'])
        except KeyError:
            self.raise_error(u'No content in a lexeme: ', dictDescr)
            return
        if len(Lexeme.obligFields & keys) < len(Lexeme.obligFields):
            self.raise_error(u'No obligatory fields in a lexeme: ',\
                             dictDescr[u'content'])
            return
        for obj in sorted(dictDescr[u'content'], key=self.fields_sorting_key):
            try:
                self.key2func[obj[u'name']](obj)
            except KeyError:
                self.add_data(obj)
        self.generate_sublexemes()

    def raise_error(self, message, data=None):
        if self.errorHandler != None:
            self.errorHandler.RaiseError(message, data)
    
    def fields_sorting_key(self, key):
        try:
            order = [u'lex', u'lexref', u'stem', u'paradigm', u'gramm',\
                     u'gloss'].index(key)
            return u'!' + unicode(order)
        except:
            return key

    def num_stems(self):
        """Return the number of different stem numbers."""
        stemNums = set([sl.numStem for sl in self.subLexemes])
        return len(stemNums)
    
    def add_lemma(self, obj):
        lemma = obj[u'value']
        if type(lemma) not in [str, unicode] or len(lemma) <= 0:
            self.raise_error(u'Wrong lemma: ', lemma)
            return
        if len(self.lemma) > 0:
            self.raise_error(u'Duplicate lemma: ' + lemma)
        self.lemma = lemma

    def add_lexref(self, obj):
        lexref = obj[u'value']
        if type(lexref) not in [str, unicode] or len(lexref) <= 0:
            self.raise_error(u'Wrong lexical reference: ', lexref)
            return
        if len(self.lexref) > 0:
            self.raise_error(u'Duplicate lexical reference: ' +\
                             lexref + u' in ' + self.lemma)
        self.lexref = lexref

    def add_stem(self, obj):
        stem = obj[u'value']
        if type(stem) not in [str, unicode] or len(stem) <= 0:
            self.raise_error(u'Wrong stem in ' + self.lemma + u': ', stem)
            return
        if len(self.stem) > 0:
            self.raise_error(u'Duplicate stem in ' + self.lemma + u': ', stem)
        self.stem = stem

    def add_gramm(self, obj):
        gramm = obj[u'value']
        if type(gramm) not in [str, unicode] or len(gramm) <= 0:
            self.raise_error(u'Wrong gramtags in ' + self.lemma + u': ', gramm)
            return
        if len(self.gramm) > 0:
            self.raise_error(u'Duplicate gramtags: ' + gramm +\
                             u' in ' + self.lemma)
        self.gramm = gramm

    def add_gloss(self, obj):
        gloss = obj[u'value']
        if type(gloss) not in [str, unicode] or len(gloss) <= 0:
            self.raise_error(u'Wrong gloss in ' + self.lemma + u': ', gloss)
            return
        if len(self.gloss) > 0:
            self.raise_error(u'Duplicate gloss: ' + gloss +\
                             u' in ' + self.lemma)
        self.gloss = gloss

    def add_paradigm(self, obj):
        paradigm = obj[u'value']
        if type(paradigm) not in [str, unicode] or len(paradigm) <= 0:
            self.raise_error(u'Wrong paradigm in ' + self.lemma +\
                             u': ', paradigm)
            return
        self.paradigms.append(paradigm)

    def add_except(self, obj):
        ex2add = ExceptionForm(obj, self.errorHandler)
        tagSet = set(ex2add.gramm.split(u','))
        try:
            if all(ex != ex2add for ex in self.exceptions[tagSet]):
                self.exceptions[tagSet].append(ex2add)
        except KeyError:
            self.exceptions[tagSet] = [ex2add]

    def add_data(self, obj):
        try:
            self.otherData.append((obj[u'name'], obj[u'value']))
        except KeyError:
            self.raise_error(u'Wrong key-value pair in ' + self.lemma +\
                             u': ', obj)

    def generate_sublexemes(self):
        self.subLexemes = []
        stems = self.separate_parts(self.stem)
        paradigms = [self.separate_parts(p) for p in self.paradigms]
        grams = self.separate_parts(self.gramm)
        glosses = self.separate_parts(self.gloss)

        # Add conversion links from the descriptions of the paradigms:
        for pGroup in paradigms:
            for p in pGroup:
                for pVariant in p:
                    try:
                        newStemConversionLinks = self.g.paradigms[pVariant].conversion_links
                        for cl in newStemConversionLinks:
                            self.otherData.append([u'conversion-link', cl])
                    except KeyError:
                        pass
        self.generate_stems(stems)
        
        if len(grams) not in [1, len(stems)]:
            self.raise_error(u'Wrong number of gramtags (' + self.gramm +\
                             u') in ' + self.lemma)
            return
        if len(glosses) not in [0, 1, len(stems)]:
            self.raise_error(u'Wrong number of glosses (' + self.gloss +\
                             u') in ' + self.lemma)
            return
        for p in paradigms:
            if len(p) not in [1, len(stems)]:
                self.raise_error(u'Wrong number of paradigms in ' +\
                                 self.lemma + u': ', p)
                return
        for iStem in range(len(stems)):
            curGloss = u''
            if len(glosses) == 1:
                curGloss = glosses[0][0] # no variants for glosses
            elif len(glosses) > 1:
                curGloss = glosses[iStem][0]
            if len(grams) == 1:
                curGramm = grams[0][0] # no variants for grams either
            elif len(grams) > 1:
                curGramm = grams[iStem][0]
            curParadigms = []
            for p in paradigms:
                if len(p) == 1:
                    curParadigms += p[0]
                else:
                    curParadigms += p[iStem]
            for curStem in stems[iStem]:
                for curParadigm in curParadigms:
                    self.subLexemes.append(SubLexeme(iStem, curStem, curParadigm,\
                                                     curGramm, curGloss))
                
    
    def separate_parts(self, s, sepParts=u'|', sepVars=u'//'):
        return [part.split(sepVars) for part in s.split(sepParts)]

    def generate_stems(self, stems):
        """Fill in the gaps in the stems description with the help of
        automatic stem conversion."""
        if self.g == None:
            return
        stemConversionNames = set(t[1] for t in self.otherData
                                  if t[0] == u'conversion-link')
        for scName in stemConversionNames:
            try:
                self.g.stemConversions[scName].convert(stems)
            except KeyError:
                self.raise_error(u'No stem conversion named ' + scName)

    def generate_redupl_paradigm(self):
        """Create new paradigms with reduplicated parts of this particular
        lexeme or change the references if they already exist."""
        if len(self.g.paradigms) <= 0:
            self.raise_error(u'Paradigms must be loaded before lexemes.')
            return
        for sl in self.subLexemes:
            if sl.paradigm not in self.g.paradigms:
                self.raise_error(u'No paradigm named ' + sl.paradigm)
                continue
            paraRedupl = self.g.paradigms[sl.paradigm].fork_redupl(sl)
            if paraRedupl.name not in self.g.paradigms:
                self.g.paradigms[paraRedupl.name] = paraRedupl
            sl.paradigm = paraRedupl.name

    def generate_regex_paradigm(self):
        """Create new paradigms where all inflexions with regexes that
        don't match to the particular stem of this lexeme are deleted
        or change the references if they already exist."""
        if len(self.g.paradigms) <= 0:
            self.raise_error(u'Paradigms must be loaded before lexemes.')
            return
        for sl in self.subLexemes:
            if sl.paradigm not in self.g.paradigms:
                self.raise_error(u'No paradigm named ' + sl.paradigm)
                continue
            paraRegex = self.g.paradigms[sl.paradigm].fork_regex(self, sl)
            if paraRegex.name not in self.g.paradigms:
                self.g.paradigms[paraRegex.name] = paraRegex
            sl.paradigm = paraRegex.name

    def generate_wordforms(self):
        """Generate a list of all possible wordforms with this lexeme."""
        if len(self.g.paradigms) <= 0:
            self.raise_error(u'Paradigms must be loaded before lexemes.')
            return
        wordforms = []
        for sl in self.subLexemes:
            if sl.paradigm not in self.g.paradigms:
                self.raise_error(u'No paradigm named ' + sl.paradigm)
                continue
            for flex in self.g.paradigms[sl.paradigm].flex:
                wf = wordform.Wordform(self, sl, flex, self.errorHandler)
                if wf.wf == None:
                    continue
                # TODO: exceptions
                wordforms.append(wf)
        return wordforms

    def add_derivations(self):
        """Add sublexemes with links to derivations."""
        subLexemes2add = []
        for sl in self.subLexemes:
            derivName = u'#deriv#paradigm#' + sl.paradigm
            if derivName in self.g.paradigms:
                slNew = copy.deepcopy(sl)
                slNew.paradigm = derivName
                subLexemes2add.append(slNew)
        self.subLexemes += subLexemes2add
        # TODO: deriv-links in the lexeme
