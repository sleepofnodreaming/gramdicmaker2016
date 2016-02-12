import re

REDUPL_SIDE_RIGHT = True
REDUPL_SIDE_LEFT = False

class RegexTest:
    def __init__(self, field, sTest, errorHandler=None):
        self.errorHandler = errorHandler
        self.field = field
        self.sTest = sTest
        try:
            self.rxTest = re.compile(self.sTest, flags=re.U)
        except:
            self.raise_error(u'Wrong regex in the test for field ' +\
                             self.field + u': ' + self.sTest)
            self.rxTest = re.compile(u'', flags=re.U)

    def raise_error(self, message, data=None):
        if self.errorHandler != None:
            self.errorHandler.RaiseError(message, data)

    def __deepcopy__(self, memo):
        newObj = RegexTest(self.field, self.sTest, self.errorHandler)
        return newObj

    def perform(self, s):
        #print 'regex: ' + s
        return self.rxTest.search(s) != None


class Replacement:
    def __init__(self, dictRepl, errorHandler=None):
        self.errorHandler = errorHandler
        self.rxWhat = None
        self.sWhat = u''
        self.sWith = u''
        if len(dictRepl[u'value']) > 0:
            self.sWhat, self.sWith = self.short_repl(dictRepl[u'value'])
        else:
            self.sWhat = u''
            self.sWith = u''
            for obj in dictRepl[u'content']:
                if obj[u'name'] == u'what':
                    self.sWhat = obj[u'value']
                elif obj[u'name'] == u'with':
                    self.sWith = obj[u'value']
                else:
                    self.raise_error(u'Unrecognized field in a replacement description: ',\
                                     repl)
        self.compile_replacement()

    def short_repl(self, s):
        m = re.search(u'^(.*?) *-> *(.*)$', s, flags=re.U)
        if m == None:
            self.raise_error(u'Wrong replacement description: ' + s)
            return u'^$', u''
        return m.group(1), m.group(2)

    def compile_replacement(self):
        try:
            self.rxWhat = re.compile(self.sWhat, flags=re.U|re.DOTALL)
        except:
            self.raise_error(u'Wrong regex in a replacement description: ' +\
                             self.sWhat)

    def convert(self, s):
        try:
            s = self.rxWhat.sub(self.sWith, s)
        except:
            self.raise_error(u'Incorrect regex in a replacement description: ',\
                             rxWhat)
        return s

    def raise_error(self, message, data=None):
        if self.errorHandler != None:
            self.errorHandler.RaiseError(message, data)

    def __deepcopy__(self, memo):
        dictDescr = {u'name': u'replace', u'value': u'',\
                     u'content': [{u'name': u'what', u'value': self.sWhat},\
                                  {u'name': u'with', u'value': self.sWith}]}
        newObj = Replacement(dictDescr, self.errorHandler)
        return newObj

class Reduplication:
    def __init__(self, arrDescr, errorHandler=None):
        self.errorHandler = errorHandler
        self.replacements = []
        self.side = REDUPL_SIDE_RIGHT
        for obj in arrDescr:
            if obj[u'name'] == u'side':
                self.change_side(obj)
            elif obj[u'name'] == u'replace':
                self.replacements.append(Replacement(obj, self.errorHandler))
            else:
                self.raise_error(u'Unrecognized field in a reduplication description: ',\
                                 obj)

    def raise_error(self, message, data=None):
        if self.errorHandler != None:
            self.errorHandler.RaiseError(message, data)

    def change_side(self, side):
        if side[u'value'] == u'right':
            self.side = REDUPL_SIDE_RIGHT
        elif side[u'value'] == u'left':
            self.side = REDUPL_SIDE_LEFT
        else:
            self.raise_error(u'Unrecognized value in a reduplication description: ',\
                             side)

    def perform(self, s):
        """Perform the reduplication on a string."""
        for repl in self.replacements:
            s = repl.convert(s)
        return s
    
            
