import codecs, os, json
import Tkinter, tkMessageBox

class ErrorHandler:
    def __init__(self):
        self.log = []
        self.logFileName = u''

    def __deepcopy__(self, memo):
        return self

    def RaiseError(self, errorMessage, data=None):
        if data != None:
            try:
                dataStr = json.dumps(data, ensure_ascii=False)
                errorMessage += dataStr
            except:
                pass
        self.log.append(errorMessage)
        window = Tkinter.Tk()
        window.wm_withdraw()
        tkMessageBox.showinfo("Error", errorMessage)

