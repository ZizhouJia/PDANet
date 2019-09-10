import threading
import httplib

class base_worker(object):
    def __init__(self,ip="127.0.0.1",port=6008):
        self.port=port
        self.task_name=task_name
        self.ip=ip
        self.conn=httplib.HTTPConnection(self.ip,self.port,timeout=500)

    def work(self,**kwargs):
        pass

    def run(self):
        try:
            status="WAIT"
            while(status=="WAIT"):
                self.conn.request('POST',"/"{})
