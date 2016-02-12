from reduplication import RegexTest, Reduplication, REDUPL_SIDE_RIGHT, REDUPL_SIDE_LEFT
from afx_table import AfxInTable, AfxTable
import lexeme
import re, copy, json

POS_NONSPECIFIED = -1
POS_NONFINAL = 0
POS_FINAL = 1
POS_BOTH = 1

GLOSS_EMPTY = 0
GLOSS_AFX = 1
GLOSS_IFX = 2
GLOSS_REDUPL_R = 3
GLOSS_REDUPL_L = 4
GLOSS_STEM = 5
GLOSS_STEM_FORCED = 6
GLOSS_STEM_SPEC = 7
GLOSS_NEXT_FLEX = 8
GLOSS_STARTWITHSELF = 100

dotDeletionRegex = re.compile(u'\\.+', flags = re.I|re.U)

class ParadigmLink:
    # The objects of this class don't allow deep copy (otherwise the derivations
    # compilation would be too resource consuming) and therefore should be
    # immutable.
    def __init__(self, dictDescr, errorHandler=None):
        self.errorHandler = errorHandler
        self.name = dictDescr[u'value']
        self.subsequent = []
        self.position = POS_NONSPECIFIED
        if u'content' not in dictDescr or dictDescr[u'content'] is None:
            return
        for obj in dictDescr[u'content']:
            if obj[u'name'] == u'paradigm':
                self.subsequent.append(ParadigmLink(obj, errorHandler))
            elif obj[u'name'] == u'position':
                self.add_position(obj)
            else:
                self.raise_error(u'Unrecognized field in a link to a paradigm: ',
                                 obj)
    
    def raise_error(self, message, data=None):
        if self.errorHandler is not None:
            self.errorHandler.RaiseError(message, data)

    def add_position(self, obj):
        v = obj[u'value']
        if v == u'final':
            self.position = POS_FINAL
        elif v == u'both':
            self.position = POS_BOTH
        elif v == u'non-final':
            self.position = POS_NONFINAL
        else:
            self.raise_error(u'Wrong position value: ', obj)

    def __deepcopy__(self, memo):
        return self


class InflexionPart:
    def __init__(self, flex, gloss, glossType):
        self.flex = flex
        self.gloss = gloss
        self.glossType = glossType


class Inflexion:
    g = None
    rxFlexSplitter = re.compile(u'(<\\.>|\\.|\\[[^\\[\\]]*\\]|[^.<>|\\[\\]]+)', flags=re.U)
    rxStemNumber = re.compile(u'^<([0-9]+)>(.*)', flags=re.U)
    rxCleanGloss = re.compile(u'[\\[\\]!~]+', flags=re.U)
    
    def __init__(self, dictDescr, errorHandler=None):
        self.flex = u''
        self.stemNum = None     # what stem it can attach to
        self.stemNumOut = None  # what stem should be able to attach to
                                # the subsequent inflexions
        self.passStemNum = True     # stemNum must be equal to stemNumOut at any time
        self.gramm = u''
        self.gloss = u''
        self.position = POS_NONSPECIFIED
        self.reduplications = {}    # number -> Reduplication object
        self.regexTests = []
        self.subsequent = []
        self.flexParts = [[]]   # list of consequently applied inflexions;
                                # when reduplications have been dealt with,
                                # this list should have length of 1
        self.errorHandler = errorHandler
        self.replaceGrammar = False     # if false, the grammar of the inflexion
                                        # is added to the grammar od the stem
                                        # or the previous inflexion
        self.keepOtherData = True   # if true, pass all the data from the lexeme
                                    # to the wordform
        self.otherData = []
        self.lemmaChanger = None    # an inflexion object which changes the lemma
        self.startWithSelf = False  # if true, start with the inflexion when joining
                                    # itself to a stem or to a previous inflexion
        try:
            self.flex = dictDescr[u'value']
        except KeyError:
            self.raise_error(u'Wrong inflexion: ', dictDescr)
            return
        # The length of the inflexion can equal zero, so we don't check for it.
        if u'content' not in dictDescr or dictDescr[u'content'] is None:
            return
        self.key2func = {u'gramm': self.add_gramm, u'gloss': self.add_gloss,
                         u'paradigm': self.add_paradigm_link,
                         u'redupl': self.add_reduplication,
                         u'lex': self.add_lemma_changer}
        for obj in dictDescr[u'content']:
            try:
                self.key2func[obj[u'name']](obj)
            except KeyError:
                if obj[u'name'].startswith(u'regex-'):
                    self.add_regex_test(obj)
                else:
                    self.add_data(obj)
        self.generate_parts()
        # Marina
        self.simpleFlextest = None # This attribute is None if the
                                # generate_simple_flextest() func wasn't called
                                # or cannot be called
        self.testParts = []
        self.regexText = u''
        self.allowSimple = False

    def raise_error(self, message, data=None):
        if self.errorHandler is not None:
            self.errorHandler.RaiseError(message, data)

    def add_gramm(self, obj):
        gramm = obj[u'value']
        if type(gramm) not in [str, unicode]:
            self.raise_error(u'Wrong gramtags in ' + self.flex + u': ', gramm)
            return
        if len(self.gramm) > 0:
            self.raise_error(u'Duplicate gramtags: ' + gramm +\
                             u' in ' + self.flex)
        self.gramm = gramm

    def add_gloss(self, obj):
        gloss = obj[u'value']
        if type(gloss) not in [str, unicode] or len(gloss) <= 0:
            self.raise_error(u'Wrong gloss in ' + self.flex + u': ', gloss)
            return
        if len(self.gloss) > 0:
            self.raise_error(u'Duplicate gloss: ' + gloss +
                             u' in ' + self.flex)
        self.gloss = gloss.replace(u'|', u'¦')

    def add_position(self, obj):
        v = obj[u'value']
        if v == u'final':
            self.position = POS_FINAL
        elif v == u'both':
            self.position = POS_BOTH
        elif v == u'non-final':
            self.position = POS_NONFINAL
        else:
            self.raise_error(u'Wrong position value: ', obj)
    
    def add_paradigm_link(self, obj, checkIfExists=False):
        if checkIfExists and any(p.name == obj[u'value']\
                                 for p in self.subsequent):
            return
        self.subsequent.append(ParadigmLink(obj, self.errorHandler))

    def add_reduplication(self, obj):
        try:
            numRedupl = int(obj[u'value'])
        except:
            self.raise_error(u'Wrong reduplication: ', obj)
            return
        if u'content' not in obj or obj[u'content'] is None:
            obj[u'content'] = []
        if numRedupl in self.reduplications:
            self.raise_error(u'Duplicate reduplication: ', obj)
        self.reduplications[numRedupl] = Reduplication(obj[u'content'],\
                                                       self.errorHandler)

    def add_regex_test(self, obj):
        if not obj[u'name'].startswith(u'regex-'):
            return
        self.regexTests.append(RegexTest(obj[u'name'][6:], obj[u'value'],\
                                         self.errorHandler))

    def add_data(self, obj):
        self.otherData.append((obj[u'name'], obj[u'value']))

    def add_lemma_changer(self, obj):
        newLemma = obj[u'value']
        if type(newLemma) not in [str, unicode]:
            self.raise_error(u'Wrong lemma in ' + self.flex + u': ', newLemma)
            return
        dictDescr = {u'name': u'flex', u'value': newLemma, u'content': []}
        self.lemmaChanger = Inflexion(dictDescr, self.errorHandler)
        self.lemmaChanger.startWithSelf = True

    def remove_stem_number(self):
        flex = self.flex
        mStemNumber = self.rxStemNumber.search(flex)
        if mStemNumber is not None:
            try:
                self.stemNum = int(mStemNumber.group(1))
                if self.stemNumOut is None:
                    self.stemNumOut = self.stemNum
                flex = mStemNumber.group(2)
            except:
                self.raise_error(u'Wrong stem number: ' + flex)
                flex = re.sub(u'^<[0-9]*>', u'', flex, flags=re.U)
        return flex
    
    def generate_parts(self):
        self.flexParts = [[]]
        flex = self.remove_stem_number()
        flexParts = self.rxFlexSplitter.findall(flex)
        if len(self.gloss) <= 0:
            glossParts = [u''] * len(flexParts)
        else:
            glossParts = self.gloss.split(u'¦')
        iGlossPart = 0
        iRedupl = 0
        bStemStarted = False
        bStemForcedRepeat = False
        for flexPart in flexParts:
            # 1. Look at the gloss.
            if u'.' not in flexPart:
                if iGlossPart >= len(glossParts):
                    self.raise_error(u'No correspondence between the inflexion ' +\
                                 u'(' + self.flex + u') and the glosses ' +\
                                 u'(' + self.gloss + u') ')
                    return
                if glossParts[iGlossPart].startswith(u'!'):
                    bStemForcedRepeat = True
                    #glossParts[iGlossPart] = glossParts[iGlossPart][1:]
                if bStemStarted and not bStemForcedRepeat:
                    glossType = GLOSS_IFX
                else:
                    glossType = GLOSS_AFX
                if len(glossParts[iGlossPart]) >= 2 and\
                   glossParts[iGlossPart][0] == u'[' and\
                   glossParts[iGlossPart][-1] == u']':
                    #glossParts[iGlossPart] = glossParts[iGlossPart][1:len(glossParts[iGlossPart])-1]
                    glossType = GLOSS_STEM_SPEC
                elif glossParts[iGlossPart].startswith(u'~'):
                    glossType = GLOSS_REDUPL_L
                    #glossParts[iGlossPart] = glossParts[iGlossPart][1:]
                elif glossParts[iGlossPart].endswith(u'~'):
                    glossType = GLOSS_REDUPL_R
                    #glossParts[iGlossPart] = glossParts[iGlossPart][:-1]
            
            # 2. Look at the inflexion.
            if len(flexPart) == 0:
                self.flexParts[0].append(InflexionPart(u'', u'', GLOSS_EMPTY))
            elif flexPart == u'0':
                self.flexParts[0].append(InflexionPart(u'', glossParts[iGlossPart],\
                                                    glossType))
                iGlossPart += 1
            elif flexPart[:2] == u'[~' and flexPart[-1] == u']':
                try:
                    m = re.search(u'^\\[~([^\\[\\]]*)\\]$', flexPart, flags=re.U)
                    if len(m.group(1)) <= 0:
                        curReduplNum = iRedupl
                        flexPart = u'[~' + unicode(curReduplNum) + u']'
                        iRedupl += 1
                    else:
                        curReduplNum = int(m.group(1))
                except:
                    self.raise_error(u'Wrong reduplication: ' + flex)
                    return
                try:
                    side = self.reduplications[curReduplNum].side
                except KeyError:
                    self.raise_error(u'No reduplication #' + str(curReduplNum) +\
                                     u': ' + flex)
                    return
                if side == REDUPL_SIDE_RIGHT:
                    glossType = GLOSS_REDUPL_R
                elif side == REDUPL_SIDE_LEFT:
                    glossType = GLOSS_REDUPL_L
                #if bStemStarted:
                bStemStarted = True
                bStemForcedRepeat = True
                self.flexParts[0].append(InflexionPart(flexPart,\
                                      glossParts[iGlossPart], glossType))
                iGlossPart += 1
            elif flexPart == u'.' or flexPart == u'[.]':
                glossType = GLOSS_STEM
                if bStemForcedRepeat:
                    glossType = GLOSS_STEM_FORCED
                elif bStemStarted:
                    glossType = GLOSS_EMPTY
                bStemStarted = True
                bStemForcedRepeat = False
                self.flexParts[0].append(InflexionPart(flexPart, u'.', glossType))
            elif flexPart[0] == u'[' and flexPart[-1] == u']':
                glossType = GLOSS_STEM
                if bStemForcedRepeat:
                    glossType = GLOSS_STEM_FORCED
                elif bStemStarted:
                    glossType = GLOSS_EMPTY
                bStemStarted = True
                bStemForcedRepeat = False
                self.flexParts[0].append(InflexionPart(flexPart[1:len(flexPart)-1],
                                                    u'', glossType))
            elif flexPart == u'<.>':
                self.flexParts[0].append(InflexionPart(u'<.>', u'<.>', GLOSS_NEXT_FLEX))
            else:
                self.flexParts[0].append(InflexionPart(flexPart,
                    self.rxCleanGloss.sub(u'', glossParts[iGlossPart]),
                                                       glossType))
                iGlossPart += 1

        self.ensure_infixes()
        self.rebuild_value()

    def ensure_infixes(self):
        """Make sure that the inflexion parts that follow the stem
        aren't called infixes."""
        for flexPartsSet in self.flexParts:
            for iFlexPart in range(len(flexPartsSet))[::-1]:
                if flexPartsSet[iFlexPart].glossType in\
                   [GLOSS_STEM, GLOSS_STEM_FORCED, GLOSS_EMPTY,
                    GLOSS_REDUPL_L, GLOSS_REDUPL_R]:
                    return
                elif flexPartsSet[iFlexPart].glossType == GLOSS_IFX:
                    flexPartsSet[iFlexPart].glossType = GLOSS_AFX

    def make_final(self):
        """Prohibit subsequent extension of the inflexion."""
        self.position = POS_FINAL
        self.subsequent = []
        if len(self.flexParts) <= 0:
            return
        self.flexParts[-1] = [part for part in self.flexParts[-1]\
                              if part.flex != u'<.>']
        self.rebuild_value()

    def rebuild_value(self):
        """Rebuild the self.flex value using the information from
        self.flexParts list.
        self.flexParts is what's responsible for the behaviour of the
        inflexion. The self.flex property can be used as a string
        representation of the inflexion, but the user must ensure
        it is up to date every time they use it."""
        newFlex = u''
        specialChars = set([u'.', u'[', u']', u'<', u'>'])
        for fps in self.flexParts:
            curFlex = u''
            if self.stemNum is not None:
                curFlex = u'<' + unicode(self.stemNum) + u'>'
            for fp in fps:
                if len(fp.flex) > 0 and len(curFlex) > 0 and\
                   fp.flex[0] not in specialChars and\
                   curFlex[-1] not in specialChars:
                    curFlex += u'|'
                curFlex += fp.flex
            if len(newFlex) > 0:
                newFlex += u' + '
            newFlex += curFlex
        self.flex = newFlex

    def simplify_redupl(self, sublex):
        """Replace [~...]'s with actual segments for the given SubLexeme."""
        if len(self.flexParts) == 1 and all(not fp.flex.startswith(u'[~')\
                                            for fp in self.flexParts[0]):
            return []
        reduplParts = []
        pTmp = Paradigm({u'name': u'paradigm', u'value': u'tmp',
                         u'content': None}, self.errorHandler)
        subLexStem = sublex.stem
        if self.startWithSelf and not subLexStem.startswith(u'.'):
            subLexStem = u'.' + subLexStem
        curStemParts = re.findall(u'(\\.|[^.]+)', subLexStem, flags=re.U)
        for iFlexPart in range(len(self.flexParts)):
            strForm = u''
            reduplNumbers = set()
            curFlexParts = [fp.flex for fp in self.flexParts[0]\
                            if fp.glossType != GLOSS_STARTWITHSELF]
            parts = [curStemParts, curFlexParts]
            pos = [0, 0] # current position in [stem, flex]
            iSide = 0 # 0 = stem, 1 = flex
            while any(pos[i] < len(parts[i]) for i in [0, 1]):
                if iSide == 0 and pos[iSide] == len(parts[iSide]):
                    iSide = 1
                elif iSide == 1 and pos[iSide] == len(parts[iSide]):
                    iSide = 0
                if parts[iSide][pos[iSide]] in [u'.', u'[.]']:
                    pos[iSide] += 1
                    if iSide == 0:
                        iSide = 1
                    elif iSide == 1:
                        if pos[1] == 1 and not pos[0] == 1:
                            continue
                        iSide = 0
                    continue
                if iSide == 1 and parts[iSide][pos[iSide]].startswith(u'[~'):
                    try:
                        m = re.search(u'^\\[~([^\\[\\]]*)\\]$',\
                                      parts[iSide][pos[iSide]], flags=re.U)
                        reduplNum = int(m.group(1))
                        reduplNumbers.add(reduplNum)
                    except:
                        self.raise_error(u'Wrong reduplication: ', fp.flex)
                strForm += parts[iSide][pos[iSide]]
                pos[iSide] += 1
            reduplParts += self.reduplicate_str(strForm, reduplNumbers)
            if len(self.flexParts) > 1:
                self.flexParts = pTmp.join_inflexion_parts([self.flexParts[0]],
                                                           self.flexParts[1:])
        self.rebuild_value()
        return reduplParts

    def reduplicate_str(self, strForm, reduplNumbers):
        reduplParts = {}
        for reduplNum in sorted(reduplNumbers):
            m = re.search(u'^(.*?)\\[~' + unicode(reduplNum) + u'\\](.*)$',
                          strForm, flags=re.U)
            if m is None:
                self.raise_error(u'Reduplication impossible: form ' + strForm +
                                 u', reduplication #' + unicode(reduplNum))
                return
            segment2reduplicate = u''
            if self.reduplications[reduplNum].side == REDUPL_SIDE_RIGHT:
                segment2reduplicate = m.group(2)
            elif self.reduplications[reduplNum].side == REDUPL_SIDE_LEFT:
                segment2reduplicate = m.group(1)
            segment2reduplicate = re.sub(u'\\[~[^\\[\\]]*\\]', u'',
                                         segment2reduplicate, flags=re.U)
            segment2reduplicate = self.reduplications[reduplNum].perform(segment2reduplicate)
            reduplParts[reduplNum] = segment2reduplicate
            strForm = m.group(1) + segment2reduplicate + m.group(2)
        self.replace_redupl_parts(reduplParts, 0)
        return [reduplParts[reduplNum] for reduplNum in sorted(reduplNumbers)]

    def replace_redupl_parts(self, reduplParts, flexPartNum=0):
        """Replace [~...]'s whose numbers are among the keys of the
        reduplParts dictionary with actual strings in the flexPart list
        with the given number."""
        if flexPartNum < 0 or flexPartNum >= len(self.flexParts):
            return
        for iFp in range(len(self.flexParts[flexPartNum])):
            fp = self.flexParts[flexPartNum][iFp]
            if fp.flex.startswith(u'[~'):
                try:
                    m = re.search(u'^\\[~([^\\[\\]]*)\\]$', fp.flex, flags=re.U)
                    reduplNum = int(m.group(1))
                    if reduplNum in reduplParts:
                        #fp.flex = reduplParts[reduplNum]
                        self.insert_redupl_part(reduplParts[reduplNum],
                                                iFp, flexPartNum)
                except:
                    self.raise_error(u'Wrong reduplication: ', fp.flex)

    def insert_redupl_part(self, reduplPart, iFp, flexPartNum):
        if flexPartNum < 0 or flexPartNum >= len(self.flexParts):
            return
        fpRedupl = self.flexParts[flexPartNum].pop(iFp)
        reduplFragmentParts = re.findall(u'(<\\.>|[^<>]+)', reduplPart, flags=re.U)
        for iReduplFragmentPart in range(len(reduplFragmentParts)):
            fpTmp = copy.deepcopy(fpRedupl)
            if reduplFragmentParts[iReduplFragmentPart] == u'<.>':
                fpTmp.gloss = u'<.>'
                fpTmp.glossType = GLOSS_NEXT_FLEX
            if iReduplFragmentPart > 1:
                fpTmp.gloss = u''
            fpTmp.flex = reduplFragmentParts[iReduplFragmentPart]
            self.flexParts[flexPartNum].insert(iFp + iReduplFragmentPart, fpTmp)

    def get_middle(self):
        """Return an Inflexion object containig only the middle parts
        (those inside the stem)."""
        flexMiddle = copy.deepcopy(self)
        flexMiddle.flexParts[0] = [fp for fp in flexMiddle.flexParts[0]
                                   if fp.glossType in [GLOSS_STEM, GLOSS_STEM_FORCED,
                                                       GLOSS_IFX, GLOSS_STEM_SPEC]]
        return flexMiddle

    def get_pfx(self):
        """Return an AfxInStem object containig the initial part of the
        inflexion (before the first stem part).
        Works correctly only if len(self.flexParts) == 1."""
        if len(self.flexParts) <= 0:
            return None
        pfx = AfxInTable()
        afxGlossed = u''
        gloss = u''
        for fp in self.flexParts[0]:
            if fp.glossType in [GLOSS_EMPTY, GLOSS_STARTWITHSELF]:
                continue
            elif fp.glossType in [GLOSS_STEM, GLOSS_STEM_FORCED,
                                  GLOSS_IFX, GLOSS_STEM_SPEC]:
                break
            pfx.afx += fp.flex
            afxGlossed += fp.flex + u'-'
            gloss += fp.gloss + u'-'
        pfx.grFeatures.add((self.gramm, u'', u'', afxGlossed, gloss, self.stemNum))
        return pfx

    def get_sfx(self):
        """Return an AfxInStem object containig the caudal part of the
        inflexion (after the last stem part).
        Works correctly only if len(self.flexParts) == 1."""
        if len(self.flexParts) <= 0:
            return None
        sfx = AfxInTable()
        afxGlossed = u''
        gloss = u''
        for fp in self.flexParts[0][::-1]:
            if fp.glossType in [GLOSS_EMPTY, GLOSS_STARTWITHSELF]:
                continue
            elif fp.glossType in [GLOSS_STEM, GLOSS_STEM_FORCED,
                                  GLOSS_IFX, GLOSS_STEM_SPEC]:
                break
            sfx.afx = fp.flex + sfx.afx
            afxGlossed = u'-' + fp.flex + afxGlossed
            gloss = u'-' + fp.gloss + gloss
        sfx.grFeatures.add((self.gramm, afxGlossed, gloss, self.stemNum))
        return sfx
                      
        
    def __unicode__(self):
        r = u'<Inflexion object>\r\n'
        r += u'flex: ' + self.flex + u'\r\n'
        r += u'gramm: ' + self.gramm + u'\r\n'
        for iFPs in range(len(self.flexParts)):
            r += u'Inflexion parts list #' + unicode(iFPs) + u'\r\n'
            for fp in self.flexParts[iFPs]:
                r += fp.flex + u'\t' + fp.gloss + u'\t' +\
                     unicode(fp.glossType) + u'\r\n'
            r += u'\r\n'
        return r

    # Marina
    def flex_part_generator(self):
        """Reorganizes an array of inflection parts, replacing all
        the sequences of empty spaces with None. The tesult is saved
        to the self.testParts variable"""
        infArr = []
        if len(self.flexParts) == 1:
            for inflexPart in self.flexParts[0]:
                if inflexPart.glossType == GLOSS_NEXT_FLEX:
                    print 'glossType failure: GLOSS_NEXT_FLEX found in a paradigm compiled'
                    raise   # Какая тут у тебя система исключений?
                            # Ну и вообще, если что, то что в таком случае делать?
                elif inflexPart.glossType == GLOSS_STARTWITHSELF:
                    pass
                elif inflexPart.glossType == GLOSS_STEM or \
                     inflexPart.glossType == GLOSS_STEM_FORCED:
                    if len(infArr) < 1 or infArr[-1] != None: # None marks (.*)
                        infArr.append(None)
                else:
                    if inflexPart.flex == 0:
                        pass
                    else:
                        if len(infArr) < 1 or infArr[-1] == None:
                            infArr.append(inflexPart.flex)
                        else:
                            infArr[-1] += inflexPart.flex
##        print '[',
##        for i in infArr:
##            print i,
##        print ']'
        self.testParts = infArr

    def compile_simple_flextest(self):
        """Generates a regex to check if there is any stem
        option in an input string and saves in to the
        self.simpleFlextest variable"""
##        if self.testParts == []:
##            self.flex_part_generator()
        reg = u''
        for i in self.testParts:
            if i == None:
                reg += u'(.*)'
            else:
                reg += i
        if reg.startswith(u'(.') and reg.endswith(u'.*)'):
            self.additionalStemTemplate = u'%s'
        elif reg.startswith(u'(.'):
            self.additionalStemTemplate = u'%s.'
        elif reg.endswith(u'.*)'):
            self.additionalStemTemplate = u'.%s'
        else:
            self.additionalStemTemplate = u'.%s.'


        reg = u'^%s$' % reg
        self.regexText = reg
##        print reg, '!'
        self.simpleFlextest = re.compile(reg, flags = re.U|re.I)
        self.numOfGroups = self.testParts.count(None)
        if len(self.testParts) == 2 and \
           (self.testParts[0] == None and self.testParts[1] != None or \
            self.testParts[0] != None and self.testParts[1] == None):
            self.allowSimple = True

    def sft2stem(self, word):
        m = self.simpleFlextest.search(word)
        stem = []
        for i in range(1, self.numOfGroups + 1):
            stem.append(m.group(i))
        if len(stem) > 1:
            stem =  u'.'.join(stem)
        else:
            stem = stem[0]
        stem = self.additionalStemTemplate % stem
        return stem

    def find_flex_parts(self, word):
        """returns an array of arrays of tuples
        made from three elements: the index of the
        first letter of an inflection part, index of the
        last """
        orderedFlexParts = [[]]
##        print 'self.testParts',
##        print '[',
##        for i in self.testParts:
##            print i,
##        print ']'
        if self.testParts:
##            print 'self.testParts:', self.testParts
            for i in self.testParts: # перебираем тестовые части
##                print orderedFlexParts, 'orderedFlexParts'
                temp = [] # temp is an array of places of inflection
                if i != None:
                    for j in range(len(word)):
##                        print u'i:', i, 'word:', word[j:]
                        if word[j:].startswith(i) == True:
                            tup = (j, j + len(i) - 1, i)
                            temp.append(tup)
                if temp != []:
##                    print 'Found: paradigm -> find_flex_parts', temp
                    tempStems = []
                    for p in orderedFlexParts: # массив, в котором лежат массивы кортежей
                        for j in temp:
##                            print j
                            tempArr = p[:]
                            tempArr.append(j)
                            tempStems.append(tempArr)
                    orderedFlexParts = tempStems
##                else:
##                    print 'Not Found: paradigm -> find_flex_parts'
                # к этому месту мы имеем собрание кусочков, где эта часть может быть
                # после чего на каждый из массивов присобачиваем по кусочку

        return orderedFlexParts

    def evaluate_flex_order(self, arrayOfTuples):
        """ Has an array of tuples looking like
        (startIndex, endIndex, textOfPart) as an arg and
        checks if the sequence of inflection parts is the same
        as in the list of flexParts """
        textTuple = tuple([i[2] for i in arrayOfTuples])
        controlTuple = tuple([i for i in self.testParts if i != None])
        if textTuple == controlTuple:
            return True
        return False

    def check_intersection(self, arrayOfTuples):
        """Returns True, if there is an intersection,
        and False, if there is not"""
        for i in range(len(arrayOfTuples) - 1):
            if arrayOfTuples[i][1] >= arrayOfTuples[i + 1][0]:
                return True
        return False

    def position_prohibited(self, arrayOfTuples, word):
        startPoint = min([i[0] for i in arrayOfTuples])
        endPoint = max([i[1] for i in arrayOfTuples])
        # if an inflection starts at the very beginning of the word
        # and the beginning should look in a different way,
        # the position is considered prohibited
        if len(self.testParts) > 0 \
           and startPoint == 0 \
           and self.testParts[0] != None \
           and self.testParts[0] != arrayOfTuples[0][2]:
            return True
        # if an inflection should be in the very end but
        # there is something else after it,
        # the combination is inappropriate, too
        if len(self.testParts) > 0 \
           and endPoint != len(word) - 1 \
           and self.testParts[-1] != None:
            return True
        return False

    def clear_flex_list(self, word):
        """ Gets the list of combinations of inflection parts
        and checks if they pass the tests, which are above.
        Returns the list of tuples."""
        flexPlaceList = self.find_flex_parts(word)
##        print flexPlaceList, u'paradigm -> clear_flex_list 1'
        flexPlaceList = [i for i in flexPlaceList \
                         if len(i) > 0 and \
                         self.evaluate_flex_order(i) == True \
                         and self.check_intersection(i) == False\
                         and self.position_prohibited(i, word) == False]
##        
##        print flexPlaceList, u'paradigm -> clear_flex_list 2'
        return flexPlaceList

    def allow_simple_search(self):
        return  self.allowSimple
    
    def possible_stems(self, word):
##        self.flex_part_generator()
##        if self.simpleFlextest == None: # защита от дурака
##            self.compile_simple_flextest()
            
        if self.simpleFlextest.search(word) != None: # есть смысл искать что-либо
            if self.allow_simple_search() == True:
##                self.sft2stem(word)
                return [self.sft2stem(word)]
            else:
                return self.possible_stems_complicated(word)
        else:
            return []

    def possible_stems_complicated(self, word):
        """ generates a set of possible stems """
    ##        print len(flexpartPositions), u'paradigm -> possible_stems'
        flexpartPositions = self.clear_flex_list(word)
        stemArray = []
        for flexKit in flexpartPositions:
            positionsToDelete = []
            for flexPart in flexKit:
                positionsToDelete += range(flexPart[0], flexPart[1] + 1)
            temp = u''
            for i in range(len(word)):
                if i not in positionsToDelete:
                    temp += word[i]
                else:
                    temp += u'.'
            stemArray.append(temp)
        stemArray = [dotDeletionRegex.sub(u'.', i) for i in stemArray]
        return list(set(stemArray))

    def possible_stem(self, word):
        """ test func to generate one stem if possible """
##        print self.__unicode__()
##        if self.simpleFlextest == None:
##            self.compile_simple_flextest()
        m = self.simpleFlextest.search(word)
        if m == None:
            return []
        stem = []
        for i in range(1, self.testParts.count(None) + 1):
            stem.append(m.group(i))
        stem = u'.'.join(stem)
##        print self.regexText
        if self.regexText.startswith(u'^(.*)') or \
           self.regexText.startswith(u'^(.*?)'):
            stem = u'.' + stem
        if self.regexText.endswith(u'(.*?)$') or \
           self.regexText.endswith(u'(.*)$'):            
            stem += u'.'
        return [stem]

    def replace_none(self, x):
        """is used in the next func"""
        if x == None:
            return u'.'
        else:
            return x

    def comparison_pattern(self):
##        if self.testParts == []:
##            self.flex_part_generator()
        """is used to search an inflection by pattern"""
        temp = [self.replace_none(i) for i in self.testParts]
        return u''.join(temp)

    def lookslike(self, what):
        """Compares the patterns for both self and arg
        Returns true, if they are equal"""
        if self.comparison_pattern() == what.comparison_pattern():
            return True
        else:
            return False

    def init_search_data(self):
        self.flex_part_generator()
        self.compile_simple_flextest()



class Paradigm:
    g = None
    rxEmptyFlex = re.compile(u'^[.<>\\[\\]0-9]*$', flags=re.U)
    
    def __init__(self, dictDescr, errorHandler=None):
        self.errorHandler = errorHandler
        self.name = dictDescr[u'value']
        self.flex = []
        self.subsequent = []
        self.derivLinks = []
        self.conversion_links = []
        self.position = POS_NONSPECIFIED
        self.regexTests = None  # (field, regex as string) -> [RegexTest,
                                # set of numbers of inflexions which rely on
                                # that regex]
                                # (the actual dictionary is built later)
        if u'content' not in dictDescr or dictDescr[u'content'] == None:
            return
        if dictDescr[u'name'] == u'paradigm':
            self.init_paradigm(dictDescr[u'content'])
        elif dictDescr[u'name'] == u'deriv-type':
            self.init_derivation(dictDescr[u'content'])
        self.redistribute_paradigms()
        #self.compile_paradigm()
    
    def raise_error(self, message, data=None):
        if self.errorHandler is not None:
            self.errorHandler.RaiseError(message, data)

    def init_derivation(self, data):
        """Create an inflexion for each stem of the derivation."""
        stems = [u'']
        glosses = [u'']
        gramms = [u'']
        newData = []
        for obj in self.separate_variants(data):
            if obj[u'name'] == u'stem':
                stems = obj[u'value'].split(u'|')
            elif obj[u'name'] == u'gloss':
                glosses = obj[u'value'].split(u'|')
            elif obj[u'name'] == u'gramm':
                gramms = obj[u'value'].split(u'|')
            else:
                newData.append(obj)
        if len(glosses) == 1 and len(stems) > 1:
            glosses *= len(stems)
        if len(gramms) == 1 and len(stems) > 1:
            gramms *= len(stems)
        if len(glosses) != len(stems) or len(gramms) != len(stems):
            self.raise_error(u'The number of glosses and grammatical tags sets ' +
                             u'should equal either 1 or the number of stems ' +
                             u'in the derivation (stem=' + u'|'.join(stems) +
                             u', gloss=' + u'|'.join(glosses) +
                             u', gramm=' + u'|'.join(gramms) + u')')
            return
        iStem = 0
        for stem, gloss, gramm in zip(stems, glosses, gramms):
            for stemVar in stem.split(u'//'):
                stemVar = re.sub(u'\\.(?!\\])', u'<.>', stemVar, flags=re.U)
                stemVar = stemVar.replace(u'[.]', u'.')
                bReplaceGrammar = True
                arrContent = copy.deepcopy(newData)
                if len(gloss) > 0:
                    arrContent.append({u'name': u'gloss', u'value': gloss})
                if gramm.startswith(u'+') or len(gramm) <= 0:
                    bReplaceGrammar = False
                    gramm = gramm[1:]
                arrContent.append({u'name': u'gramm', u'value': gramm})
                dictDescr = {u'name': u'flex', u'value': stemVar,
                             u'content': arrContent}
                flex = Inflexion(dictDescr, self.errorHandler)
                flex.passStemNum = False
                if len(stems) > 1:
                    flex.stemNumOut = iStem
                flex.position = POS_NONFINAL
                flex.replaceGrammar = bReplaceGrammar
                flex.keepOtherData = False
                flex.startWithSelf = True
                if len(flex.flexParts[0]) > 0:
                    flex.flexParts[0].insert(0, InflexionPart(u'', u'',
                                                              GLOSS_STARTWITHSELF))
                self.flex.append(flex)
            iStem += 1
        
    def init_paradigm(self, data):
        for obj in self.separate_variants(data):
            if obj[u'name'] == u'flex':
                self.flex.append(Inflexion(obj, self.errorHandler))
            elif obj[u'name'] == u'paradigm':
                self.subsequent.append(obj)
            elif obj[u'name'] == u'position':
                self.position = obj[u'value']
            elif obj[u'name'] == u'deriv-link':
                self.add_deriv_link(obj)
            elif obj[u'name'] == u'conversion-link':
                self.conversion_links.append(obj[u'value'])
            else:
                self.raise_error(u'Unrecognized field in a paradigm: ' +
                                 obj[u'name'])

    def add_deriv_link(self, obj):
        self.derivLinks.append(obj)
    
    def separate_variants(self, arrDescr):
        for obj in arrDescr:
            if obj[u'name'] != u'flex' or u'/' not in obj[u'value']:
                yield obj
            else:
                values = obj[u'value'].split(u'//')
                for value in values:
                    objVar = copy.deepcopy(obj)
                    objVar[u'value'] = value
                    yield objVar
        
    def redistribute_paradigms(self):
        """Copy paradigm-level links to subsequent paradigms to each
        individual inflexion."""
        if self.position != POS_NONSPECIFIED:
            for flex in self.flex:
                if flex.position == POS_NONSPECIFIED:
                    flex.position = self.position
        for obj in self.subsequent:
            for flex in self.flex:
                flex.add_paradigm_link(obj, True)
        self.subsequent = []
        self.position = POS_NONSPECIFIED

    def build_regex_tests(self):
        """Build a dictionary which contains all regex tests from
        the inflexions.
        Must be performed after the paradigm has been compiled."""
        self.regexTests = {}
        for iFlex in range(len(self.flex)):
            flex = self.flex[iFlex]
            for rt in flex.regexTests:
                if rt.field == u'next':
                    continue
                sField, sRx = rt.field, rt.sTest
                if sField == u'prev':
                    sField = u'stem'
                try:
                    self.regexTests[(sField, sRx)][1].add(iFlex)
                except KeyError:
                    self.regexTests[(sField, sRx)] = [rt, set([iFlex])]

    def compile_paradigm(self):
        """Recursively join all the inflexions with the subsequent ones."""
        depth = 0
        # each inflexion can join non-empty subsequent inflexions
        # at most self.g.DERIV_LIMIT times.
        for f in self.flex:
            f.join_depth = 1
        while any((f.position != POS_FINAL and
                   f.join_depth < self.g.DERIV_LIMIT)
                  for f in self.flex) and\
                depth <= self.g.TOTAL_DERIV_LIMIT:
            newFlex = []
            newFlexExtensions = []
            for f in self.flex:
##                if self.name == u'#deriv#paradigm#Nctt':
##                    print depth, f.flex
##                    print unicode(f)
                if depth == 0:
                    shortName = re.sub(u'#paradigm#[^#]+$', u'',
                                       self.name, flags=re.U)
                    f.dictRecurs = {shortName: 1}
                    # dictRecurs is a temporary dictionary which shows
                    # how many times certain paradigms were used in the
                    # course of this inflexion's generation
                if len(f.subsequent) <= 0 or f.position == POS_FINAL or\
                   f.position == POS_BOTH:
                    fNew = copy.deepcopy(f)
                    fNew.make_final()
                    fNew.__dict__.pop('dictRecurs', None)
                    newFlex.append(fNew)
                    if len(f.subsequent) <= 0 or f.position == POS_FINAL:
                        continue
                if f.join_depth >= self.g.DERIV_LIMIT:
                    continue
                newFlexExtensions += self.extend_one(f)
            self.flex = newFlex + newFlexExtensions
            if len(newFlexExtensions) <= 0:
                break
            depth += 1
        self.remove_redundant()

    def remove_redundant(self):
        """Remove 'hanging', i. e. strictly non-final, inflexions
        from the list of inflexions after the compilation of the paradigm."""
        for iFlex in range(len(self.flex))[::-1]:
            f = self.flex[iFlex]
            if len(f.subsequent) > 0 and f.position != POS_FINAL and\
                            f.position != POS_BOTH:
                self.flex.pop(iFlex)
            else:
                f.__dict__.pop('dictRecurs', None)

    def extend_one(self, flexL):
        extensions = []
        # if u'гӕрз' in flexL.flex:
        #     print unicode(flexL)
        #     print len(flexL.subsequent), u'****************************'
        for paradigmLink in flexL.subsequent:
            shortName = re.sub(u'#paradigm#[^#]+$', u'',
                               paradigmLink.name, flags=re.U)
            # if u'гӕрз' in flexL.flex:
            #     print shortName
            dictRecurs = flexL.dictRecurs.copy()
            try:
                dictRecurs[shortName] += 1
            except KeyError:
                dictRecurs[shortName] = 1
            # if u'гӕрз' in flexL.flex:
            #     print dictRecurs[shortName]
            if dictRecurs[shortName] > self.g.RECURS_LIMIT:
                continue
##            print unicode(len(self.g.paradigms[paradigmLink.name].flex)) +\
##                  u' inflexions to be added.'
            for flexR in self.g.paradigms[paradigmLink.name].flex:
                #if self.name == u'#deriv#paradigm#Nctt':
                # if u'гӕрз' in flexL.flex or u'гӕрз' in flexR.flex:
                #     print unicode(flexL)
                #     print u'+'
                #     print unicode(flexR)
                #     print u'------------'
##                        print len(extensions)
                flexExt = self.join_inflexions(copy.deepcopy(flexL),
                                               copy.deepcopy(flexR),
                                               copy.deepcopy(paradigmLink))
                if flexExt is not None:
                    # if u'гӕрз' in flexL.flex or u'гӕрз' in flexR.flex:
                    #     print unicode(flexExt)
                    flexExt.dictRecurs = dictRecurs
                    # the same dictRecurs is used for all resulting inflexions
                    # of this step
                    extensions.append(flexExt)
        return extensions

    def join_inflexions(self, flexL, flexR, paradigmLink):
        #print flexL.flex, flexR.flex
        if not self.stem_numbers_agree(flexL, flexR):
            return None
        if not self.join_regexes(flexL, flexR):
            return None

        # Manage links to the subsequent paradigms:
        if paradigmLink.position != POS_NONSPECIFIED:
            flexL.position = paradigmLink.position
        else:
            flexL.position = flexR.position
        if paradigmLink.position == POS_FINAL:
            flexL.make_final()
        elif len(paradigmLink.subsequent) > 0:
            flexL.subsequent = paradigmLink.subsequent
        else:
            flexL.subsequent = flexR.subsequent

        # Join all other fields:
        if flexR.replaceGrammar == False:
            if len(flexL.gramm) > 0 and len(flexR.gramm) > 0:
                flexL.gramm += u','
            flexL.gramm += flexR.gramm
        else:
            flexL.gramm = flexR.gramm
            flexL.replaceGrammar = True
        if flexR.keepOtherData == False:
            flexL.keepOtherData = False
        self.join_reduplications(flexL, flexR)
        flexL.flexParts = self.join_inflexion_parts(flexL.flexParts,
                                                    flexR.flexParts)
        flexL.ensure_infixes()
        flexL.rebuild_value()
        #print u'Result: ' + flexL.flex
        return flexL

    def stem_numbers_agree(self, flexL, flexR):
        """Check if the inflexions' stem number fields agree.
        Make both stem numbers equal.
        Return True if the numbers agree, and False if they don't."""
        if flexL.stemNumOut is not None and flexR.stemNum is not None and\
           flexL.stemNumOut != flexR.stemNum:
            return False
        if flexL.stemNumOut is None or flexL.passStemNum == True:
            flexL.stemNumOut = flexR.stemNumOut
            if flexL.stemNum is None or flexL.passStemNum == True:
                if flexR.stemNum != None:
                    flexL.stemNum = flexR.stemNum
                else:
                    flexR.stemNum = flexL.stemNum
            flexL.passStemNum = flexL.passStemNum or flexR.passStemNum
        elif flexR.stemNumOut is not None and flexR.passStemNum == False:
            flexL.stemNumOut = flexR.stemNumOut
        return True

    def flex_is_empty(self, flexValue):
        """Check if the inflexion does not contain any non-empty segments."""
        if self.rxEmptyFlex.search(flexValue):
            return True
        return False

    def join_regexes(self, flexL, flexR):
        """Check if the inflexions' regexes agree.
        If they agree, add flexR's regexes to flexL and return True,
        if they don't, return False."""
        bAgree = True
        flexL.rebuild_value()
        valueL = flexL.flex
        if len(flexL.flexParts) > 1:
            valueL = re.sub(u'^.* + ', u'', valueL)
        flexR.rebuild_value()
        valueR = flexR.flex
        if len(flexR.flexParts) > 1:
            valueR = re.sub(u' + .*', u'', valueR)
        for rxNext in flexL.regexTests:
            if rxNext.field == u'next' and rxNext.perform(valueR) is None:
                return False

        bEmptyL = self.flex_is_empty(valueL)
        bEmptyR = self.flex_is_empty(valueR)
        if not (bEmptyL or bEmptyR):
            flexL.join_depth += 1
        # If the left inflexion is empty, regex-prev of the right inflexion
        # become regex-stem of the joined inflexion.
        tests2add = []
        for rxPrev in flexR.regexTests:
            if rxPrev.field == u'prev':
                if bEmptyL:
                    if all(rt.field != u'stem' or rt.sTest != rxPrev.sTest\
                           for rt in flexL.regexTests):
                        tests2add.append(copy.deepcopy(rxPrev))
                        tests2add[-1].field = u'stem'
                else:
                    if rxPrev.perform(valueL) == False:
                        return False
            elif rxPrev.field.startswith(u'prev-'):
                field2test = rxPrev.field[5:]
                if field2test == u'gramm' and not rxPrev.perform(flexL.gramm):
                    return False
                elif field2test == u'gloss' and not rxPrev.perform(flexL.gloss):
                    return False
            elif all(rt.field != rxPrev.field or rt.sTest != rxPrev.sTest\
                     for rt in flexL.regexTests):
                tests2add.append(copy.deepcopy(rxPrev))
        for rxTest in tests2add:
            flexL.regexTests.append(rxTest)
        return True
    
    def join_inflexion_parts(self, flexPartsL, flexPartsR):
        if any((fp.glossType == GLOSS_REDUPL_L and fp.flex.startswith(u'[~')) or
               (fp.glossType == GLOSS_REDUPL_R and fp.flex.startswith(u'[~'))
               for fp in flexPartsL[-1]):
            return flexPartsL + flexPartsR
        
        if len(flexPartsL[-1]) <= 0:
            return flexPartsL[:-1] + flexPartsR
        elif len(flexPartsR[0]) <= 0:
            return flexPartsL + flexPartsR[1:]

        if flexPartsR[0][0].glossType == GLOSS_STARTWITHSELF:
            fpOldR = flexPartsR[0][1:]
            if flexPartsL[-1][0].glossType == GLOSS_STARTWITHSELF:
                fpOldL = flexPartsL[-1][1:]
            else:
                fpOldL = flexPartsL[-1]
            if fpOldL[0].flex != u'<.>':
                fpOldL.insert(0, InflexionPart(u'<.>', u'<.>', GLOSS_NEXT_FLEX))
            fpNew = [InflexionPart(u'', u'', GLOSS_STARTWITHSELF)]
        else:
            fpOldR = flexPartsR[0]
            if flexPartsL[-1][0].glossType == GLOSS_STARTWITHSELF:
                fpOldL = flexPartsL[-1][1:]
            else:
                fpOldL = flexPartsL[-1]
            fpNew = [InflexionPart(u'', u'', GLOSS_STARTWITHSELF)]
        
        fpOld = [fpOldL, fpOldR]
        pos = [0, 0]
        iSide = 0
        bStemStarted = False
        bStemForcedRepeat = False
        while any(pos[i] < len(fpOld[i]) for i in [0, 1]):
            if iSide == 0 and pos[iSide] == len(fpOld[iSide]):
                iSide = 1
            elif iSide == 1 and pos[iSide] == len(fpOld[iSide]):
                iSide = 0
            if iSide == 0 and\
               fpOld[iSide][pos[iSide]].glossType == GLOSS_NEXT_FLEX:
                pos[iSide] += 1
                iSide = 1
                continue
            elif iSide == 1 and fpOld[iSide][pos[iSide]].flex == u'.':
                if fpOld[iSide][pos[iSide]].glossType == GLOSS_STEM_FORCED:
                    bStemForcedRepeat = True
                if pos[1] == 0 and not pos[0] == 1:
                    pos[iSide] += 1
                    continue
                pos[iSide] += 1
                iSide = 0
                continue
            elif fpOld[iSide][pos[iSide]].glossType == GLOSS_STARTWITHSELF:
                pos[iSide] += 1
                continue
            fp = InflexionPart(fpOld[iSide][pos[iSide]].flex,
                               fpOld[iSide][pos[iSide]].gloss,
                               fpOld[iSide][pos[iSide]].glossType)
            if not bStemStarted and fp.glossType == GLOSS_IFX:
                fp.glossType = GLOSS_AFX
            elif fp.glossType in [GLOSS_STEM, GLOSS_STEM_FORCED, GLOSS_EMPTY]:
                if bStemForcedRepeat or fp.glossType == GLOSS_STEM_FORCED:
                    fp.glossType = GLOSS_STEM_FORCED
                    bStemForcedRepeat = False
                elif not bStemStarted:
                    fp.glossType = GLOSS_STEM
                else:
                    fp.glossType = GLOSS_EMPTY
                bStemStarted = True
            elif fp.glossType in [GLOSS_REDUPL_L, GLOSS_REDUPL_R]:
                bStemStarted = True
            elif bStemStarted and fp.glossType == GLOSS_AFX:
                fp.glossType = GLOSS_IFX
            pos[iSide] += 1
            fpNew.append(fp)
        return flexPartsL[:-1] + [fpNew] + flexPartsR[1:]

    def join_reduplications(self, flexL, flexR):
        """Renumber the reduplications in flexR."""
        if len(flexL.reduplications) <= 0 or\
           len(flexR.reduplications) <= 0:
            return
        maxReduplNumL = max(flexL.reduplications.keys())
        dictNewReduplR = {}
        for k, v in flexR.reduplications.iteritems():
            dictNewReduplR[k + 1 + maxReduplNumL] = v
        flexR.reduplications = dictNewReduplR
        for fps in flexR.flexParts:
            for fp in fps:
                if fp.GlossType in [GLOSS_REDUPL_R, GLOSS_REDUPL_L]:
                    try:
                        m = re.search(u'^\\[~([^\\[\\]]*)\\]$',
                                      fp.flex, flags=re.U)
                        reduplNum = int(m.group(1)) + 1 + maxReduplNumL
                        fp.flex = u'[~' + unicode(reduplNum) + u']'
                    except:
                        self.raise_error(u'Wrong reduplication: ', fp.flex)

    def fork_redupl(self, sublex):
        """Return a reduplication-free version of self."""
        newPara = copy.deepcopy(self)
        reduplParts = []
        for flex in newPara.flex:
            reduplParts += flex.simplify_redupl(sublex)
        if len(reduplParts) > 0:
            newPara.name += u'~' + u'~'.join(reduplParts)
        return newPara

    def fork_regex(self, lex, sublex):
        """Return a regex-free version of self."""
        if self.regexTests is None:
            self.build_regex_tests()
        newPara = copy.deepcopy(self)
        if len(newPara.regexTests) == 0:
            return newPara
        testResults = 0
        flex2remove = set()
        for rtKey in sorted(newPara.regexTests):
            result = lexeme.check_for_regex(lex, sublex,
                                            newPara.regexTests[rtKey][0],
                                            self.errorHandler)
            if result == False:
                flex2remove |= newPara.regexTests[rtKey][1]
            testResults = testResults * 2 + int(result)
        for iFlex in sorted(flex2remove, reverse=True):
            newPara.flex.pop(iFlex)
        newPara.name += u'=' + unicode(testResults)
        newPara.regexTests = {}
        for flex in newPara.flex:
            newPara.regexTests = []
        return newPara

    def get_pfx(self, linkName):
        """Return the table of all possible prefixes in this paradigm."""
        pfxTable = AfxTable()
        for flex in self.flex:
            flexPfx = flex.get_pfx()
            flexPfx.links.add(linkName)     # the link to the stems table
            pfxTable.add(flexPfx)
        return pfxTable

    def get_sfx(self):
        """Return the table of all possible suffixes in this paradigm."""
        sfxTable = AfxTable()
        for flex in self.flex:
            flexSfx = flex.get_sfx()
            sfxTable.add(flexSfx)
        return sfxTable

    def get_middle(self):
        """Return the table of inflexions equal to the middle parts
        of the inflexions of the current paradigm."""
        middleTable = AfxTable()
        for flex in self.flex:
            flexMiddle = flex.get_middle()
            middleTable.add_flex(flexMiddle)
        return middleTable

    def get_stems(self, lexemes, linkName):
        """Return the table of all versions of stems for the given lexemes,
        with infixes inserted in them."""
        middleTable = self.get_middle()
        middleInflexions = middleTable.afxs.values()
        stemTable = AfxTable()
        for lex, sl in lexemes:
            for middleFlex in middleInflexions:
                stemInTable = sl.make_stem(lex, middleFlex)
                if stemInTable:
                    stemInTable.links.add(linkName)    # the link to the sfx table
                    stemTable.add(stemInTable)
                    #print stemInTable.afx, stemInTable.grFeatureSetLink, stemInTable.lexFeatureSetLink
        return stemTable



