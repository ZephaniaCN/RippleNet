from logging import getLogger
import nni
logger = getLogger()
class Reporters():
    def __init__(self, eval_baseline):
        reporter_init_dict = {
            'nni': NniReporter,
            'normal': Reporter,
        }

    def intermediate_report(self, eval_res, test_res, epoch):
        if eval_res:
            for key, value in eval_res.items():
                self.reporters['eval'][key].intermediate_report(value, epoch)
        if test_res:
            for key, value in eval_res.items():
                self.reporters['test'][key].intermediate_report(value, epoch)



class Reporter():
    def __init__(self, name):
        self.best_res=0
        self.name = name
    def intermediate_report(self,res, epoch):
        logger.info('{}:{}\tepoch:{}'.format(self.name, res, epoch))
        if (res > self.best_res):
            self.best_res = res
    def fin_report(self):
        logger.info('finished\nbest res:\n{}:{}'.format(self.name, self.best_res))
        best_res = self.best_res
        return best_res
class NniReporter():
    def __init__(self):
        self.best_res=0
    def intermediate_report(self, res, epoch):
        nni.report_intermediate_result(res)
        if(res>self.best_res):
            self.best_res = res
    def fin_report(self):
        nni.report_final_result(self.best_res)
        best_res=self.best_res
        return best_res