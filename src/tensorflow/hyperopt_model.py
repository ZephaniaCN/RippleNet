
from concurrent import futures
from hyperopt import Trials
from src.tensorflow.train import  run_exp
from src.tensorflow.args import args_convert, ripple_net_plus_book_args
import time
import logging
logger = logging.getLogger()
bayes_trials = Trials()



bayes_trials = Trials()




def model_thread(args):
    exector = futures.ProcessPoolExecutor(max_workers=1)
    f = exector.submit(run_exp,args_convert(args))
    while True:
        time.sleep(100)
        if f.exception():
            logging.debug('exception:',f.exception())
            logging.debug('check thread alive')
            try:
                exector.shutdown(wait=False)
            except:
                pass
            exector = futures.ProcessPoolExecutor(max_workers=1)
            args['batch_size']=int(args['batch_size']/2)
            args['n_epoch'] += 10
            logger.info('batch_size:{}\nepoch:{}'.format(args['batch_size'],args['n_epoch']))
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

    args = {**ripple_net_plus_book_args, **hyperparameters}
    args['file_name']='{}_{:.1e}lr_{:.1e}kg_{:.1e}l2_{:.2f}dp_{}m_{}h_{}d_{}'.format(
        args['file_name'],args['lr'],args['kge_weight'],args['l2_weight'],args['dropout'],
        args['n_memory'],args['n_hop'],args['dim'],args['predict_mode']
    )
    res_dict = model_thread(args)

    return 1-res_dict['auc']

