
from concurrent import futures
from hyperopt import Trials
from src.tensorflow.train import  run_exp
from src.tensorflow.args import args_convert, ripple_net_plus_args
import time
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
bayes_trials = Trials()



bayes_trials = Trials()




def model_thread(args):
    exector = futures.ProcessPoolExecutor(max_workers=1)
    f = exector.submit(run_exp,args_convert(args))
    while True:
        time.sleep(10)
        if f.exception():
            logging.debug('check thread alive')
            try:
                exector.shutdown(wait=False)
            except:
                pass
            exector = futures.ProcessPoolExecutor(max_workers=1)
            logger.debug('chage batch_size and ')
            args['batch_size']=int(args['batch_size']/2)
            args['n_epoch'] += 10

            f = exector.submit(run_exp,args_convert(args))
            continue
        if f.done() and not f.exception():
            res = f.result()
            logger.debug('res:{}'.format(res))
            exector.shutdown()
            break
    return res
int_list=['n_hop','n_memory','dim']

def objective(hyperparameters):
    for key in int_list:
        hyperparameters[key] = int(hyperparameters[key])

    args = {**ripple_net_plus_args, **hyperparameters}
    args['filename']='{}'.format(args['filename'],)
    res_dict = model_thread(args)

    return 1-res_dict['acc']

