import codecs, re, os

def read_file(fname, errorHandler=None):
    f = codecs.open(fname, 'r', 'utf-8-sig')
    lines = [re.sub(u'#.*', u'', line.strip(u'\r\n'))\
                 for line in f.readlines()]
    f.close()

    arr, numLine = process_lines(lines, errorHandler)
    return arr


def process_lines(lines, errorHandler=None, startFrom=0, prevOffset=0):
    if startFrom == len(lines):
        return []
    #print u'Function call'
    arr = []
    i = startFrom
    while i < len(lines):
        line = lines[i]
        #print line
        if re.search(u'^\\s*$', lines[i]) != None:
            i += 1
            continue
        m = re.search(u'^( *)(-?)([^ :]+)((?::.*)?)$', line, flags=re.U)
        if m == None:
            if errorHandler != None:
                errorHandler.RaiseError(u'Line #' + str(i) + u' is wrong: ' +\
                                        line)
            i += 1
            continue
        if len(m.group(1)) < prevOffset:
            #print u'Return'
            return arr, i
        elif len(m.group(1)) > prevOffset:
            if errorHandler != None:
                errorHandler.RaiseError(u'Wrong offset in line #' + str(i) +\
                                        u': ' + line)
            i += 1
            continue
        obj = {}
        obj[u'name'] = m.group(3)
        
        # "-lexeme" vs. "-paradigm: N1"
        if m.group(4) != None:
            obj[u'value'] = re.sub(u'^: *| *$', u'', m.group(4))
        else:
            obj[u'value'] = u''

        # "-paradigm: N1" vs. "gramm: N"
        if len(m.group(2)) == 0:
            obj[u'content'] = None
            i += 1
        else:
            obj[u'content'], i = process_lines(lines, errorHandler,\
                                               i+1, prevOffset+1)
        arr.append(obj)
    return arr, len(lines)


