import logging
from tensorboardX import SummaryWriter
import nni
logger = logging.getLogger()
class Reporters():
    def __init__(self,mode, if_nni, nni_key):
        reporter_init_dict = {
            'tensorboard': TensorboardReporter,
            'print': PrintReporter,
        }
        self.reporter = reporter_init_dict[mode]()
        self.if_nni=if_nni
        if if_nni:
            self.nni_reporter = NniReporter(nni_key)
    def intermediate_report(self, res:dict, epoch):
        self.reporter.intermediate_report(res,epoch)
        if self.if_nni:
            self.nni_reporter.intermediate_report(res)


    def fin_report(self):
        self.reporter.fin_report()
        if self.if_nni:
            self.nni_reporter.fin_report()




class PrintReporter():
    def __init__(self):
        self.logger = logging.getLogger()
    def intermediate_report(self, res:dict, epoch):
        for key, data in res.items():
            for phase, value in data.items():
                self.report(phase, key, value, epoch)

    def report(self, phase, key, value, epoch):
        self.logger.info('{}/{} {}:{}'.format(phase, key,value,epoch))
    def fin_report(self):
        pass


class TensorboardReporter():
    def __init__(self):
        self.writer = SummaryWriter()
    def intermediate_report(self, res:dict, epoch):
        logging.info('{},{}'.format(res,epoch))
        for key, data in res.items():
            self.writer.add_scalars(key,data,epoch)

    def fin_report(self):
        self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()


class NniReporter():
    def __init__(self, nni_key):
        self.best_res=0
        self.nni_key = nni_key
    def intermediate_report(self, res):
        nni.report_intermediate_result(res[self.nni_key]['eval'])
        if(res>self.best_res):
            self.best_res = res
    def fin_report(self):
        nni.report_final_result(self.best_res)
        best_res=self.best_res
        return best_res