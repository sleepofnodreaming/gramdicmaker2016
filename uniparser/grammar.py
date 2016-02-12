from ErrorHandler import ErrorHandler
from lexeme import Lexeme
from stem_conversion import StemConversion
from paradigm import Inflexion, Paradigm
from derivations import Derivation
from afx_table import AfxInTable, AfxTable, FeatureSetLink, write_feature_sets
import derivations
import yamlReader
import json, copy, sys, gc, codecs

class Grammar:
    """The main class of the project."""

    RECURS_LIMIT = 2
    DERIV_LIMIT = 4             # counts only non-empty derivands
    TOTAL_DERIV_LIMIT = 10      # counts everything
    LEX_MEMORY_LIMIT = 1024 * 1024 * 1024 # Memory limit for the lexeme list
    PARADIGM_MEMORY_LIMIT = 1024 * 1024 * 1024 # Memory limit for the paradigm list
    
    def __init__(self, errorHandler=None):
        self.lexemes = []
        self.paradigms = {}         # name -> Paradigm object
        self.lexByParadigm = {}     # paradigm name -> links to sublexemes which
                                    # have that paradigm in the form (lex, subLex)
        self.afxTables = {}         # name -> AfxTable object
        self.pfxTable = AfxTable()  # the single prefix table
        self.pfxTable.name = u'pfx'
        self.stemConversions = []
        self.derivations = {}
        if errorHandler is None:
            self.errorHandler = ErrorHandler()
        else:
            self.errorHandler = errorHandler
        Lexeme.g = self
        Inflexion.g = self
        Paradigm.g = self
        Derivation.g = self

    def load_stem_conversions(self, fname):
        """Load stem conversion rules from a file.
        Return the number of rules loaded."""
        if len(self.lexemes) > 0:
            self.raise_error(u'Loading stem conversions should occur before ' +\
                             u'loading stems.')
            return 0
        conversionDescrs = yamlReader.read_file(fname, self.errorHandler)
        self.stemConversions = {} # {conversion name -> StemConversion}
        for dictSC in conversionDescrs:
            sc = StemConversion(dictSC, self.errorHandler)
            self.stemConversions[sc.name] = sc
        return len(self.stemConversions)

    def load_paradigms(self, fname, pLst=None):
        """Load paradigms from a file.
        Return the number of paradigms loaded."""
        if len(self.lexemes) > 0:
            self.raise_error(u'Loading paradigms should occur before ' +\
                             u'loading stems.')
            return 0
        paraDescrs = yamlReader.read_file(fname, self.errorHandler)
        for dictDescr in paraDescrs:
            if sys.getsizeof(self.paradigms) > self.PARADIGM_MEMORY_LIMIT:
                self.raise_error(u'Not enough memory for the paradigms.')
                return
            self.paradigms[dictDescr[u'value']] =\
                                    Paradigm(dictDescr, self.errorHandler)
        newParadigms = {}
        for pName, p in self.paradigms.iteritems():
            if pLst is None or pName in pLst:
                p = copy.deepcopy(p)
                p.compile_paradigm()
                newParadigms[pName] = p
        self.paradigms = newParadigms
        return len(self.paradigms)

    def load_lexemes(self, fname):
        lexDescrs = yamlReader.read_file(fname, self.errorHandler)
        for dictDescr in lexDescrs:
            if sys.getsizeof(self.lexemes) > self.LEX_MEMORY_LIMIT:
                self.raise_error(u'Not enough memory for the lexemes.')
                return
            self.lexemes.append(Lexeme(dictDescr, self.errorHandler))
        return len(self.lexemes)

    def load_derivations(self, fname):
        """Load derivations from a file.
        Return the number of derivations loaded."""
        derivDescrs = yamlReader.read_file(fname, self.errorHandler)
        for dictDescr in derivDescrs:
##            self.derivations[u'#deriv#' + dictDescr[u'value']] =\
##                                    Paradigm(dictDescr, self.errorHandler)
            dictDescr[u'value'] = u'#deriv#' + dictDescr[u'value']
            self.derivations[dictDescr[u'value']] =\
                                    Derivation(dictDescr, self.errorHandler)
        for paradigm in self.paradigms.values():
            derivations.deriv_for_paradigm(paradigm)
        for derivName, deriv in self.derivations.iteritems():
            if derivName.startswith(u'#deriv#paradigm#'):
                deriv.build_links()
                print derivName + u': build complete.'
                #print unicode(self.derivations[u'#deriv#paradigm#Nctt'])
                deriv.extend_leaves()
                print derivName + u': leaves extended.'
                #print unicode(deriv)
        #print unicode(self.derivations[u'#deriv#N-fӕ#paradigm#Nct'])
        print u'Leaves extended.'
        #print unicode(self.derivations[u'#deriv#paradigm#Nct'])
        
        for derivName, deriv in self.derivations.iteritems():
            p = deriv.to_paradigm()
            self.paradigms[derivName] = p
        for derivName in self.derivations:
            print u'Compiling ' + derivName + u'... ',
            self.paradigms[derivName].compile_paradigm()
            print u'compiled.'
            gc.collect()
            if derivName == u'#deriv#paradigm#Nctt':
                fPara = codecs.open(u'test-ossetic/deriv-Nctt-test.txt', 'w', 'utf-8-sig')
                for f in self.paradigms[derivName].flex:
                    fPara.write(unicode(f))
                fPara.close()
        print u'Derivations compiled.'
        for lex in self.lexemes:
            lex.add_derivations()
        return len(self.derivations)

    def compile_all(self):
        for lex in self.lexemes:
            lex.generate_redupl_paradigm()
            lex.generate_regex_paradigm()
            if sys.getsizeof(self.paradigms) > self.PARADIGM_MEMORY_LIMIT:
                self.raise_error(u'Not enough memory for the paradigms.')
                return
            for sl in lex.subLexemes:
                try:
                    self.lexByParadigm[sl.paradigm].append((lex, sl))
                except KeyError:
                    self.lexByParadigm[sl.paradigm] = [(lex, sl)]

    def build_afx_tables(self):
        """Build the tables of prefixes, stems+infixes, and suffixes."""
        for paraName in self.paradigms:
            try:
                curLexemes = self.lexByParadigm[paraName]
                tableStems = self.paradigms[paraName].get_stems(curLexemes,
                                                                u'sfx=' + paraName)
                tableStems.name = u'stems=' + paraName
                #print u'Stem table: ' + tableStems.name
            except KeyError:
                continue    # no lexemes with that paradigm
            tablePfx = self.paradigms[paraName].get_pfx(tableStems.name)
            self.pfxTable += tablePfx
            tableSfx = self.paradigms[paraName].get_sfx()
            tableSfx.name = u'sfx=' + paraName
            self.afxTables[tableStems.name] = tableStems
            self.afxTables[tableSfx.name] = tableSfx

    def clean_unused_feature_sets(self):
        """Empty all feature sets in the FeatureSetLink class for which
        there is no active link."""
        usedGrSets = set()
        usedLexSets = set()
        usedLinkSets = set()
        for t in self.afxTables.values():
            for afx in t.afxs.values():
                for fsl in afx.fsLinks:
                    usedGrSets.add(fsl.grSetNum)
                    usedLexSets.add(fsl.lexSetNum)
                    usedLinkSets.add(fsl.linkSetNum)
        for afx in self.pfxTable.afxs.values():
            for fsl in afx.fsLinks:
                usedGrSets.add(fsl.grSetNum)
                usedLexSets.add(fsl.lexSetNum)
                usedLinkSets.add(fsl.linkSetNum)
        for iFS in range(len(FeatureSetLink.grFeatureSets)):
            if iFS not in usedGrSets:
                FeatureSetLink.grFeatureSets[iFS] = set()
        for iFS in range(len(FeatureSetLink.lexFeatureSets)):
            if iFS not in usedLexSets:
                FeatureSetLink.lexFeatureSets[iFS] = set()
        for iFS in range(len(FeatureSetLink.linkSets)):
            if iFS not in usedLinkSets:
                FeatureSetLink.linkSets[iFS] = set()
    
    def write_afx_tables(self, fname):
        # Delete the contents of the file
        f = codecs.open(fname, 'w', 'utf-8-sig')
        f.close()
        self.clean_unused_feature_sets()
        write_feature_sets(fname)
        self.pfxTable.write(fname)
        for t in self.afxTables:
            self.afxTables[t].write(fname)
        


