import logging
from tensorboardX import SummaryWriter
import nni
class Reporters():
    def __init__(self,mode, if_nni, nni_key):
        reporter_init_dict = {
            'tensorboard': TensorboardReporter,
            'print': PrintReporter,
        }
        self.reporter = reporter_init_dict[mode]()
        self.if_nni=if_nni
        if(if_nni):
            self.nni_reporter = NniReporter()
        self.nni_key = nni_key
    def intermediate_report(self, res:dict, epoch):
        for phase, data in res.items():
            for key, value in data.items():
                self.reporter.report(phase, key, value, epoch)
                if(phase=='eval' and key==self.nni_key):
                    self.nni_reporter.intermediate_report(value)

    def fin_report(self):
        if(self.if_nni):
            self.nni_reporter.fin_report()




class PrintReporter():
    def __init__(self):
        self.logger = logging.getLogger()
    def report(self, phase, key, value, epoch):
        self.logger.info('{}/{}'.format(phase, key), value, epoch)


class TensorboardReporter():
    def __init__(self):
        self.writer = SummaryWriter('log')

    def report(self,phase,key,value , epoch):
        self.writer.add_scalar('{}/{}'.format(phase,key), value, epoch)


class NniReporter():
    def __init__(self):
        self.best_res=0
    def intermediate_report(self, res):
        nni.report_intermediate_result(res)
        if(res>self.best_res):
            self.best_res = res
    def fin_report(self):
        nni.report_final_result(self.best_res)
        best_res=self.best_res
        return best_res