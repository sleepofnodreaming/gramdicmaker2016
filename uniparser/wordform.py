import lexeme
from paradigm import InflexionPart, Inflexion, Paradigm
import paradigm
import re, copy

rxClean = re.compile(u'([>~\\-])-+|-+([<~])$', flags=re.U)


def join_stem_flex(stem, stemGloss, flex, bStemStarted=False):
    wfGlossed = u''
    gloss = u''
    wf = u''
    pfxPart = u''
    ifxs = u''
    mainPart = u''
    curStemParts = re.findall(u'(\\.|[^.]+)', stem, flags=re.U)
    curFlexParts = flex.flexParts[0]
    stemSpecs = u''.join([u'.' + fp.gloss for fp in curFlexParts\
                          if fp.glossType == paradigm.GLOSS_STEM_SPEC])
    parts = [curStemParts, curFlexParts]
    pos = [0, 0]    # current position in [stem, flex]
    iSide = 0       # 0 = stem, 1 = flex
    glossType = paradigm.GLOSS_STEM
    while any(pos[i] < len(parts[i]) for i in [0, 1]):
        if iSide == 0 and pos[iSide] == len(parts[iSide]):
            iSide = 1
        elif iSide == 1 and pos[iSide] == len(parts[iSide]):
            iSide = 0
        if (iSide == 0 and parts[iSide][pos[iSide]] in [u'.', u'[.]']) or\
           (iSide == 1 and parts[iSide][pos[iSide]].flex in [u'.', u'[.]']):
            pos[iSide] += 1
            if iSide == 0:
                iSide = 1
            elif iSide == 1:
                if pos[1] == 1 and not pos[0] == 1:
                    continue
                glossType = parts[iSide][pos[iSide] - 1].glossType
                iSide = 0
            continue
        elif iSide == 1 and\
             parts[iSide][pos[iSide]].glossType == paradigm.GLOSS_STARTWITHSELF:
            pos[iSide] += 1
            continue
        curPart = parts[iSide][pos[iSide]]
        if iSide == 0:
            wf += curPart
            bStemStarted = True
            wfGlossed += curPart
            if glossType in [paradigm.GLOSS_STEM, paradigm.GLOSS_STEM_FORCED]:
                mainPart += stemGloss + stemSpecs
        elif iSide == 1:
            wf += curPart.flex.replace(u'0', u'∅')
            if curPart.glossType == paradigm.GLOSS_AFX:
                if bStemStarted:
                    mainPart += u'-' + curPart.gloss + u'-'
                else:
                    pfxPart += u'-' + curPart.gloss + u'-'
                wfGlossed += u'-' + curPart.flex + u'-'
            elif curPart.glossType == paradigm.GLOSS_IFX:
                ifxs += u'<' + curPart.gloss + u'>'
                wfGlossed += u'<' + curPart.flex + u'>'
            elif curPart.glossType == paradigm.GLOSS_REDUPL_R:
##                    if bStemStarted:
                bStemStarted = True
                mainPart += u'-' + curPart.gloss + u'~'
##                    else:
##                        pfxPart += u'-' + curPart.gloss + u'~'
                wfGlossed += u'-' + curPart.flex + u'~'
            elif curPart.glossType == paradigm.GLOSS_REDUPL_L:
##                    if bStemStarted:
                bStemStarted = True
                mainPart += u'~' + curPart.gloss + u'-'
##                    else:
##                        pfxPart += u'~' + curPart.gloss + u'-'
                wfGlossed += u'~' + curPart.flex + u'-'
            elif curPart.glossType == paradigm.GLOSS_STEM_SPEC:
                wfGlossed += curPart.flex
            elif curPart.glossType in [paradigm.GLOSS_STEM,
                                       paradigm.GLOSS_STEM_FORCED]:
                bStemStarted = True
                wfGlossed += curPart.flex
                mainPart += stemGloss + stemSpecs
            elif curPart.glossType == paradigm.GLOSS_EMPTY:
                bStemStarted = True
                wfGlossed += curPart.flex
        pos[iSide] += 1
        gloss = pfxPart + ifxs + mainPart
        gloss = rxClean.sub(u'\\1', gloss).strip(u'-~')
        wfGlossed = rxClean.sub(u'\\1', wfGlossed).strip(u'-~')
    return wf, wfGlossed, gloss


class Wordform:
    def __init__(self, lex, sublex, flex, errorHandler=None):
        self.errorHandler = errorHandler
        self.wf = None
        self.wfGlossed = u''
        self.gloss = u''
        self.lemma = u''
        self.otherData = []
        if flex.stemNum is not None and 1 < lex.num_stems() <= flex.stemNum:
            self.raise_error(u'Incorrect stem number: lexeme ' +
                             lex.lemma + u', inflexion ' +
                             flex.flex)
            return
        elif flex.stemNum is None and lex.num_stems() > 1:
            self.raise_error(u'Unspecified stem number: lexeme ' +\
                             lex.lemma + u', inflexion ' +\
                             flex.flex)
            return
        elif len(flex.flexParts) > 1:
            self.raise_error(u'The inflexion ' + flex.flex +\
                             u' is not fully compiled.')
            return
        elif lexeme.check_compatibility(lex, sublex, flex) == False:
            return
        self.add_gramm(sublex, flex)
        self.build_value(lex, sublex, flex)
        self.add_lemma(lex, flex)
        self.add_other_data(lex, flex)
        self.otherData = copy.deepcopy(lex.otherData)

    def raise_error(self, message, data=None):
        if self.errorHandler != None:
            self.errorHandler.RaiseError(message, data)

    def add_lemma(self, lex, flex):
        if flex.lemmaChanger is None:
            self.lemma = lex.lemma
            return
        suitableSubLex = [sl for sl in lex.subLexemes
                          if flex.lemmaChanger.stemNum is None or
                             sl.numStem == flex.lemmaChanger.stemNum]
        if len(suitableSubLex) <= 0:
            if len(set(sl.numStem for sl in lex.subLexemes)) == 1:
                suitableSubLex = lex.subLexemes
        if len(suitableSubLex) <= 0:
            self.raise_error(u'No stems available to create the new lemma ' +
                             flex.lemmaChanger.flex)
            self.lemma = u''
            return
        if len(suitableSubLex) > 1:
            self.raise_error(u'Several stems available to create the new lemma ' +
                             flex.lemmaChanger.flex)
        wfLemma = Wordform(lex, suitableSubLex[0], flex.lemmaChanger,
                           self.errorHandler)
        self.lemma = wfLemma.wf

    def add_gramm(self, sublex, flex):
        if flex.replaceGrammar == False:
            self.gramm = sublex.gramm
            if len(sublex.gramm) > 0 and len(flex.gramm) > 0:
                self.gramm += u','
            self.gramm += flex.gramm
        else:
            self.gramm = flex.gramm
    
    def add_other_data(self, lex, flex):
        if flex.keepOtherData:
            self.otherData = copy.deepcopy(lex.otherData)
            return

    def get_lemma(self, lex, flex):
        # TODO: lemma changers
        self.lemma = lex.lemma

    def build_value(self, lex, sublex, flex):
        subLexStem = sublex.stem
        if flex.startWithSelf and not subLexStem.startswith(u'.'):
            subLexStem = u'.' + subLexStem
        self.wf, self.wfGlossed, self.gloss = join_stem_flex(subLexStem,
                                                             sublex.gloss,
                                                             flex)

    def __unicode__(self):
        r = u'<Wordform object>\r\n'
        if self.wf is not None:
            r += self.wf + u'\r\n'
        r += self.lemma + u'; ' + self.gramm + u'\r\n'
        r += self.wfGlossed + u'\r\n'
        r += self.gloss + u'\r\n'
        return r
