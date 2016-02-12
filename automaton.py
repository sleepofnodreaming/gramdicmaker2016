""" Module containing a custom finite automaton-like thing.

"""

class FlexAutomaton(object):
    """ A base class of an automaton looking for inflection options in a token.

    """
    def __init__(self, empty):
        """ Initialize an instance. Args:
         empty - a marker of a arbitrary non-empty sequence of symbols should be used in flex templates.

         """
        self.marker = empty
        self.root = None # a node to start with
        self.entities = 0

    def add(self, flex):
        """ Add an inflection to an automaton as a search option. The places stem parts may be attached to should be marked wih the same symbol as is saved in self.marker.

        """
        self.entities += 1
        if self.root is None:
            self.root = FFANode(marker=self.marker)
        self.root.add_branch(flex)

    def parse(self, token):
        """ Parse a token into an inflection and a stem; return a list of options possible.
        If the token contains self.marker, an AssertionError will be raised.

        """
        assert self.marker not in token
        if self.root:
            return self.root.parse(token)
        return []



class FFANode(object):

    def __init__(self, marker, parent=None):
        self._dot = marker

        self.paths = {} # 'a letter read now' -> next node

        self.lastRead = None
        self.terminalState = False # boolean: is a terminal state or not. This flag does not mean there's no node's children.
        # These two variables are relevant is the state is terminal:
        self.absEnd = False # if it is true, a line being checked should end in this node.
        self.notAbsEnd = False # Alternatively, the line needn't end here.
        self.parentNode = parent
        self.flex = {}


    def add_branch(self, flex, pointer=-1):
        """ Add a slice of an inflection [0:pointer+1] to a subtree. An order letters are added in is inverted.

        """
        if pointer == -len(flex)-1:
            self.terminalState = True
            self.flex['abs'] = flex
            self.absEnd = True
            return

        if (pointer == -len(flex)) and flex[0] == self._dot:
            self.terminalState = True
            self.flex['not'] = flex
            self.notAbsEnd = True
            return

        if flex[pointer] not in self.paths:
            newNode = FFANode(parent=self, marker=self._dot)
            newNode.lastRead = flex[pointer]
            self.paths[flex[pointer]] = newNode

        self.paths[flex[pointer]].add_branch(flex, pointer-1)

    def _handle_shifts(self, token, indexFromEnd):
        """ Get all the options of terminals for all the strings from the null one to [0:indexFromEnd+1].

        """
        options = []
        for i in xrange(-len(token), indexFromEnd+1):
            options += self.paths[self._dot]._find(token, i)
        return options

    def _to_stem(self, token, suffix, indices):
        """ Given a token, a flex found and a list of indices flex parts found in, return a stem.
        Indices point to LAST elements of the parts, negative.

        """
        if suffix == self._dot:
            return token
        pts = [i for i in suffix.split(self._dot) if i]
        line = u''
        index, suff = -1, -1
        while index >= -len(token):
            if suff < -len(indices) or indices[suff] < index:
                line = token[index] + line
                index -= 1
            elif indices[suff] == index:
                line = self._dot + line
                index -= len(pts[suff])
                suff -=1
        return line

    def _find(self, token, indexFromEnd=-1):
        """ Given a token and a shift from the end, return a list of options (flex, list of indices). Indices point to the places where the relevant inflexion parts END.

        """

        if not (len(token) + indexFromEnd < 0) and token[indexFromEnd] in self.paths:
            options = self.paths[token[indexFromEnd]]._find(token, indexFromEnd-1)
        else: # if it is impossible to go further, recursion stops
            options = []
        if self.terminalState and ((not self.absEnd ^ (len(token) + indexFromEnd == -1)) or self.absEnd and self.notAbsEnd):
            if len(token) + indexFromEnd == -1:
                instance = [self.flex['abs'], []]
            else:
                instance = [self.flex['not'], []]
            # print ''
            options.append(instance)

        if indexFromEnd == -1:
            for option in options:
                option[1].append(indexFromEnd)

        if self._dot in self.paths:
            options += self._handle_shifts(token, indexFromEnd-1)

        if self.lastRead == self._dot:
            for option in options:
                option[1].append(indexFromEnd)

        return options

    def parse(self, token):
        options = self._find(token)
        return [(flex, self._to_stem(token, flex, indices)) for flex, indices in options]


if __name__ == '__main__':
    f = FlexAutomaton(empty=u'#')
    f.add(u'#mj')
    t = u'umj'
    o = f.parse(t)
    print o
