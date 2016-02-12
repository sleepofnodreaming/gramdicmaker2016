import copy, json, re
from paradigm import Paradigm


def deriv_for_paradigm(paradigm):
    """Generate a Derivation object for the given paradigm."""
    derivLinks = {}     # recurs_class -> set of Derivation names
    maxRecursClass = 0
    #print u'\n\n\n' + paradigm.name
    for derivLink in paradigm.derivLinks:
        recursClass, derivLink = get_recurs_class(derivLink)
        #print recursClass, derivLink[u'value']
        if maxRecursClass < recursClass:
            maxRecursClass = recursClass
        pName = fork_deriv(derivLink, paradigm.name)
        if len(pName) > 0:
            try:
                derivLinks[recursClass].add(pName)
            except KeyError:
                derivLinks[recursClass] = set([pName])
    handle_recurs_classes(derivLinks, maxRecursClass)
    unifiedDerivContent = []
    for derivNamesSet in derivLinks.values():
        for derivName in derivNamesSet:
            unifiedDerivContent.append({u'name': u'paradigm',
                                        u'value': derivName,
                                        u'content': []})
    if len(unifiedDerivContent) <= 0:
        return
    unifiedName = u'#deriv#paradigm#' + paradigm.name
    unifiedDeriv = Derivation({u'name': u'deriv-type', u'value': unifiedName,
                               u'content': unifiedDerivContent},
                              Derivation.g.errorHandler)
    Derivation.g.derivations[unifiedName] = unifiedDeriv


def fork_deriv(derivLink, paradigmName):
    """Create a new derivation with customized properties on the basis
    of an existing one.
    Return the name of the resulting derivation."""
    derivName = derivLink[u'value']
    try:
        newDeriv = copy.deepcopy(Derivation.g.derivations[u'#deriv#' +
                                                          derivName])
    except KeyError:
        Derivation.g.raise_error(u'No derivation named ' + derivName)
        return u''
    existingParadigms = newDeriv.find_property(u'paradigm')
    if len(existingParadigms) <= 0:
        newDeriv.add_property(u'paradigm', paradigmName)
    if derivLink[u'content'] is not None:
        for propName in set([obj[u'name'] for obj in derivLink[u'content']]):
            newDeriv.del_property(propName)
        for obj in derivLink[u'content']:
            newDeriv.add_property(obj[u'name'], obj[u'value'])
    newDerivName = newDeriv.dictDescr[u'value'] + u'#paradigm#' + \
        paradigmName
    newDeriv.dictDescr[u'value'] = newDerivName
    Derivation.g.derivations[newDerivName] = newDeriv
    return newDerivName


def get_recurs_class(derivLink):
    """Find the recurs_class property in the contents.
    Return its value and the dictionary with recurs_value removed."""
    recursClass = 0
    if derivLink[u'content'] is None or len(derivLink[u'content']) <= 0:
        return 0, derivLink
    newDerivLink = copy.deepcopy(derivLink)
    for iObj in range(len(newDerivLink[u'content']))[::-1]:
        obj = newDerivLink[u'content'][iObj]
        if obj[u'name'] == u'recurs_class':
            try:
                recursClass = int(obj[u'value'])
            except:
                Derivation.g.raise_error(u'Incorrect recurs_class value: ' +
                                         obj[u'value'])
            newDerivLink[u'content'].pop(iObj)
    return recursClass, newDerivLink


def handle_recurs_classes(derivLinks, maxRecursClass):
    """For every derivation in the dictionary, add links to the derivations
    with recurs_class less than recurs_class of that derivation."""
    links = []
    restrictedDerivs = set([re.sub(u'#paradigm#[^#]+$', u'', dv)
                            for s in derivLinks.values() for dv in s])
    prevDerivLinks = set()
    for recursClass in range(maxRecursClass + 1):
        try:
            curDerivLinks = derivLinks[recursClass]
            restrictedDerivs -= set([re.sub(u'#paradigm#[^#]+$', u'', dv)
                                     for dv in prevDerivLinks])
            curRestrictedDerivs = copy.deepcopy(restrictedDerivs)
            prevDerivLinks = curDerivLinks
        except KeyError:
            #print u'No recurs_class ' + unicode(recursClass)
            continue
        linksExtension = []
        for derivName in curDerivLinks:
            try:
                deriv = Derivation.g.derivations[derivName]
            except KeyError:
                Derivation.g.raise_error(u'No derivation named ' + derivName)
                continue
            for link in links:
                deriv.add_dict_property(link)
            deriv.restrictedDerivs = curRestrictedDerivs
            if recursClass < maxRecursClass:
                newLink = {u'name': u'paradigm', u'value': derivName,
                           u'content': [copy.deepcopy(p)
                                        for p in deriv.find_property(u'paradigm')]}
                for link in links:
                    newLink[u'content'].append(copy.deepcopy(link))
                linksExtension.append(newLink)
        links += linksExtension


def add_restricted(recursCtr, restrictedDerivs):
    recursCtr = recursCtr.copy()
    for rd in restrictedDerivs:
        recursCtr[rd] = Derivation.g.RECURS_LIMIT + 1
    return recursCtr


def extend_leaves(data, sourceParadigm, recursCtr=None,
                  removeLong=False, depth=0):
    # recursCtr: derivation name -> number of times it has been used
    if recursCtr is None:
        recursCtr = {}
    depth += 1
    data2add = []
    #print json.dumps(recursCtr, indent=1)
    #print len(recursCtr), max([0] + recursCtr.values())
    for iObj in range(len(data))[::-1]:
        obj = data[iObj]
        if obj[u'name'] != u'paradigm':
            continue
        elif obj[u'value'].startswith(u'#deriv#'):
            shortName = re.sub(u'#paradigm#[^#]+$', u'',
                               obj[u'value'], flags=re.U)
            try:
                recursCtr[shortName] += 1
            except KeyError:
                recursCtr[shortName] = 1
            if recursCtr[shortName] > Derivation.g.RECURS_LIMIT or \
                    depth > Derivation.g.DERIV_LIMIT:
                if removeLong:
                    data.pop(iObj)
                continue
            ##            print u'recurs: ' + json.dumps(obj[u'content'], ensure_ascii=False,\
            ##                                           indent=1)
            try:
                deriv = Derivation.g.derivations[obj[u'value']]
            except KeyError:
                continue
            recursCtrNext = add_restricted(recursCtr, deriv.restrictedDerivs)
            #if len(deriv.restrictedDerivs) > 0:
                #print json.dumps(list(deriv.restrictedDerivs), ensure_ascii=False)
                #print json.dumps(recursCtrNext, ensure_ascii=False)
                #print (u'*' * 20 + u'\n') * 10
                #print deriv.dictDescr[u'value'], len(deriv.restrictedDerivs)
            extend_leaves(obj[u'content'], sourceParadigm,
                          recursCtrNext, removeLong, depth)
        else:
            #print obj[u'value']
            if depth > Derivation.g.DERIV_LIMIT or \
                    obj[u'value'] == sourceParadigm:
                continue
            try:
                deriv = Derivation.g.derivations[u'#deriv#paradigm#' +
                                                 obj[u'value']]
            except KeyError:
                continue
            subsequentDerivs = copy.deepcopy(deriv.find_property(u'paradigm'))
            #print json.dumps(subsequentDerivs, indent=1)
            recursCtrNext = add_restricted(recursCtr, deriv.restrictedDerivs)
            extend_leaves(subsequentDerivs, sourceParadigm,
                          recursCtrNext, True, depth)
            data2add += subsequentDerivs
    data += data2add


class Derivation:
    """An auxiliary class where derivations are represented by dictionaries.
    After priorities are handled, all derivations should be transformed into
    paradigms."""

    g = None

    def __init__(self, dictDescr, errorHandler=None):
        self.dictDescr = copy.deepcopy(dictDescr)
        if self.dictDescr[u'content'] is None:
            self.dictDescr[u'content'] = []
        self.errorHandler = errorHandler
        self.restrictedDerivs = set()

    def raise_error(self, message, data=None):
        if self.errorHandler is not None:
            self.errorHandler.RaiseError(message, data)

    def content(self):
        return self.dictDescr[u'content']

    def find_property(self, propName):
        return [el for el in self.content() if el[u'name'] == propName]

    def add_property(self, name, value):
        self.dictDescr[u'content'].append({u'name': name, u'value': value,
                                           u'content': []})

    def add_dict_property(self, dictProperty):
        self.dictDescr[u'content'].append(copy.deepcopy(dictProperty))

    def del_property(self, propName):
        for iObj in range(len(self.dictDescr[u'content']))[::-1]:
            obj = self.dictDescr[u'content'][iObj]
            if obj[u'name'] == propName:
                self.dictDescr[u'content'].pop(iObj)

    def __unicode__(self):
        return json.dumps(self.dictDescr, ensure_ascii=False, indent=2)

    def build_links(self):
        """Add the links from all subsequent derivations to self."""
        newDerivLinks = []
        for derivLink in self.find_property(u'paradigm'):
            if (not derivLink[u'value'].startswith(u'#deriv#')) or \
                    (derivLink[u'content'] is not None and
                     len(derivLink[u'content']) > 0):
                newDerivLinks.append(derivLink)
                continue
            newDerivLink = copy.deepcopy(derivLink)
            try:
                targetDeriv = self.g.derivations[newDerivLink[u'value']]
            except KeyError:
                self.raise_error(u'No derivation named ' + newDerivLink[u'value'])
                continue
            newDerivLink[u'content'] = \
                copy.deepcopy(targetDeriv.find_property(u'paradigm'))
            newDerivLinks.append(newDerivLink)
        self.del_property(u'paradigm')
        for newDerivLink in newDerivLinks:
            self.add_dict_property(newDerivLink)

    def extend_leaves(self):
        """For the leaves in the subsequent derivation tree, which are
        real paradigms, add their subsequent derivations, if needed."""
        m = re.search(u'#deriv#paradigm#([^#]+$)', self.dictDescr[u'value'],
                      flags=re.U)
        if m is None:
            return
        paradigmName = m.group(1)
        recursCtr = {}
        for derivName in self.restrictedDerivs:
            recursCtr[derivName] = self.g.RECURS_LIMIT + 1
        extend_leaves(self.dictDescr[u'content'], paradigmName, recursCtr)

    def to_paradigm(self):
        """Create a paradigm from self.dictDescr and return it."""
        return Paradigm(self.dictDescr, self.errorHandler)
        
        
        

    
