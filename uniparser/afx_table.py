from ErrorHandler import ErrorHandler
import json, copy, sys, codecs

def unicode_special(value):
    if value is None:
        return u'-1'
    return unicode(value)


def write_feature_sets(fname):
    """Write out all the feature sets stored in the AfxTable class."""
    # The file is opened for appending
    f = codecs.open(fname, 'a', 'utf-8-sig')
    f.write(u'<gfs>\r\n')
    for iFS in range(len(FeatureSetLink.grFeatureSets)):
        f.write(u'<fs id="' + str(iFS) + u'">\r\n')
        f.write(u''.join(u'<f>' + u'\t'.join(map(unicode_special, feature)) + u'</f>\r\n'
                         for feature in FeatureSetLink.grFeatureSets[iFS]))
        f.write(u'</fs>\r\n')
    f.write(u'</gfs>\r\n<lfs>\r\n')
    for iFS in range(len(FeatureSetLink.lexFeatureSets)):
        f.write(u'<fs id="' + str(iFS) + u'">\r\n')
        f.write(u''.join(u'<f>' + u'\t'.join(map(unicode_special, feature)) + u'</f>\r\n'
                         for feature in FeatureSetLink.lexFeatureSets[iFS]))
        f.write(u'</fs>\r\n')
    f.write(u'</lfs>\r\n<ls>')
    for iFS in range(len(FeatureSetLink.linkSets)):
        f.write(u'<fs id="' + str(iFS) + u'">\r\n')
        f.write(u''.join(u'<f>' + link + u'</f>\r\n'
                         for link in FeatureSetLink.linkSets[iFS]))
        f.write(u'</fs>\r\n')
    f.write(u'</ls>\r\n')
    f.close()


class FeatureSetLink:
    """Each affix can have a number of different combinations of the set of
    grammatical features, the set of lexical features, and the set of links
    to the subsequent affix tables. Each combination is stored as an object
    of FeatureSetLink class."""
    grFeatureSets = []      # list of sets of feature tuples (common for all tables)
    lexFeatureSets = []     # list of sets of feature tuples (common for all tables)
    linkSets = []           # list of sets of links (common for all tables)
    
    def __init__(self, afx=None):
        self.grSetNum = None
        self.lexSetNum = None
        self.linkSetNum = None
        if afx is not None:
            self.grSetNum = self.add_gr_feature_set(afx.grFeatures)
            self.lexSetNum = self.add_lex_feature_set(afx.lexFeatures)
            self.linkSetNum = self.add_link_set(afx.links)
            afx.grFeatures = set()
            afx.lexFeatures = set()
            afx.links = set()

    def add_gr_feature_set(self, featureSet):
        try:
            return FeatureSetLink.grFeatureSets.index(featureSet)
        except:
            FeatureSetLink.grFeatureSets.append(featureSet)
            return len(FeatureSetLink.grFeatureSets) - 1

    def add_lex_feature_set(self, featureSet):
        try:
            return FeatureSetLink.lexFeatureSets.index(featureSet)
        except:
            FeatureSetLink.lexFeatureSets.append(featureSet)
            return len(FeatureSetLink.lexFeatureSets) - 1

    def add_link_set(self, linkSet):
        try:
            return FeatureSetLink.linkSets.index(linkSet)
        except:
            FeatureSetLink.linkSets.append(linkSet)
            return len(FeatureSetLink.linkSets) - 1

    def get_gr_fs(self):
        """Return the grammatical feature set the current object points to."""
        try:
            return FeatureSetLink.grFeatureSets[self.grSetNum]
        except KeyError:
            return set()

    def get_lex_fs(self):
        """Return the lexical feature set the current object points to."""
        try:
            return FeatureSetLink.lexFeatureSets[self.lexSetNum]
        except KeyError:
            return set()

    def get_link_set(self):
        """Return the link set the current object points to."""
        try:
            return FeatureSetLink.linkSets[self.linkSetNum]
        except KeyError:
            return set()

    def append(self, other):
        """If 2 out of 3 feature sets of self and other coincide, add the
        contents of the remaining feature set to that of self and return True,
        otherwise return False."""
        if self == other:
            return True
        if self.grSetNum == other.grSetNum and\
           self.lexSetNum == other.lexSetNum:
            fsNew = self.get_link_set() | other.get_link_set()
            self.linkSetNum = self.add_link_set(fsNew)
            return True
        if self.grSetNum == other.grSetNum and\
           self.linkSetNum == other.linkSetNum:
            fsNew = self.get_lex_fs() | other.get_lex_fs()
            self.lexSetNum = self.add_lex_feature_set(fsNew)
            return True
        if self.linkSetNum == other.linkSetNum and\
           self.lexSetNum == other.lexSetNum:
            fsNew = self.get_gr_fs() | other.get_gr_fs()
            self.grSetNum = self.add_gr_feature_set(fsNew)
            return True
        return False

    def __eq__(self, other):
        if self.grSetNum == other.grSetNum and\
           self.lexSetNum == other.lexSetNum and\
           self.linkSetNum == other.linkSetNum:
            return True
        return False

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.grSetNum, self.lexSetNum, self.linkSetNum))

    def __unicode__(self):
        return u';'.join((unicode_special(self.grSetNum),
                          unicode_special(self.lexSetNum),
                          unicode_special(self.linkSetNum)))


class AfxInTable:
    """A single affix or stem (with infixes) to be inserted into a table."""
    def __init__(self):
        self.afx = u''
        # the following 3 properties are auxiliary and should be transferred
        # to self.fsLinks at some point before the table is written to file
        self.grFeatures = set()   # set of tuples (gramm, afxGlossed, gloss, stemNum)
        self.lexFeatures = set()  # set of tuples (lex, grdic, afxGlossed, gloss, stemNum)
        self.links = set()      # set of names of the subsequent table
        
        self.fsLinks = set()    # set of FeatureSetLink objects

    def add_values(self, afx):
        """Add feature and link sets from another affix with the same string
        representation."""
        curFSLinks = afx.fsLinks
        if len(curFSLinks) <= 0:
            fsl = FeatureSetLink(afx)
            curFSLinks = set([fsl])
        for fsl in curFSLinks:
            fslAppended = False
            for fslTmp in self.fsLinks:
                if fslTmp.append(fsl):
                    fslAppended = True
                    break
            if not fslAppended:
                self.fsLinks.add(fsl)

    def __unicode__(self):
        s = u'<afx>\r\n<s>' + self.afx + u'</s>\r\n'
        s += u''.join(u'<fsl>' + unicode(fsl) + u'</fsl>\r\n'
                      for fsl in self.fsLinks)
        s += u''.join(u'<gf>' + u'\t'.join(map(unicode_special, feature)) + u'</gf>\r\n'
                      for feature in self.grFeatures)
        s += u''.join(u'<lf>' + u'\t'.join(map(unicode_special, feature)) + u'</lf>\r\n'
                      for feature in self.lexFeatures)
        s += u''.join(u'<l>' + link + u'</l>\r\n'
                      for link in self.links)
        s += u'</afx>\r\n'
        return s


class AfxTable:
    """A table of affixes or stems (with infixes)."""

    def __init__(self):
        self.name = u''
        self.afxs = {}      # affix as a string -> AfxInTable objects

    def add(self, afx):
        """Add an AfxInTable object to the table."""
        try:
            afxExisting = self.afxs[afx.afx]
            afxExisting.add_values(afx)
        except KeyError:
            # add the affix and transfer its features to the feature set lists
            self.afxs[afx.afx] = afx
            if len(afx.fsLinks) <= 0:
                # use the sets in afx
                fsl = FeatureSetLink(afx)
                afx.fsLinks.add(fsl)

    def add_flex(self, flex):
        """Add an Inflexion object (such objects should be replaced with
        AfxInTable objects later)."""
        curFeatures = set([(flex.gramm, u'', u'', flex.stemNum)])
        try:
            flex.rebuild_value()
            afxExisting = self.afxs[flex.flex]
            afxExisting.grFeatures |= curFeatures
        except KeyError:
            afx2add = AfxInTable()
            afx2add.afx = flex
            afx2add.grFeatures = curFeatures
            self.afxs[flex.flex] = afx2add

    def write(self, fname):
        # The file is opened for appending
        f = codecs.open(fname, 'a', 'utf-8')
        f.write(u'<table name="' + self.name.replace(u'"', u'&quote;') + u'">\r\n')
        for afx in sorted(self.afxs, key = lambda afx: (len(afx), afx)):
            f.write(unicode(self.afxs[afx]))
        f.write(u'</table>\r\n')
        f.close()

    def __iadd__(self, other):
        for afx in other.afxs.values():
            self.add(afx)
        return self

    def __radd__(self, other):
        if other == None:
            return copy.deepcopy(self)
        return None
        
        
