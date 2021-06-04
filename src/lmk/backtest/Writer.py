import time

class Writer():
    def __init__(self, screen = True, ds = ''):
        self.buffer = ""
        self.screen = screen
        self.ds = ds


    def print(self, mystring):
        self.buffer = self.buffer + '\n' + mystring


    def write(self):
        if self.screen:

            print(self.buffer)
        else:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            file = open("./analyses/log_" + timestr + '_'+ self.ds+'.txt', "w")
            file.write(self.buffer)
            file.close()