
import kernel as k
from threading import Thread

def main():
    for i in xrange(0,10):
        t = Thread(target=sma, args=())
        t.start()