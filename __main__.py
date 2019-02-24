
import numpy as np
from src.tensorflow.hyperopt_model import objective
from hyperopt import hp,fmin,tpe,Trials
from hyperopt.mongoexp import MongoTrials
import logging
import sys
logger = logging.getLogger()
logger.setLevel(logging.INFO)
sys.path.append('../..')
np.random.seed(555)

MAX_EVALS=500

space = {
    'n_hop': hp.choice('n_hop', [2,3,4]),
    'dim':hp.quniform('dim',2,128,1),
    'kge_weight': hp.loguniform('kge_weight',np.log(1e-9),np.log(10)),
    'l2_weight': hp.loguniform('l2_weight',np.log(1e-9),np.log(10)),
    'n_memory':hp.quniform('n_memory',2,128,1),
    'lr':hp.loguniform('lr',np.log(1e-5),np.log(1)),
    'dropout':hp.uniform('dropout',0.5,1),
    'batch_size':hp.choice('batch_size',[128,256,384,512,640,768,886,1024,1152]),
    'predict_mode':hp.choice("predict_mode",['dense','basic'])
}

trials=Trials()
#trials = MongoTrials('mongo://localhost:27017/mnn_db/jobs', exp_key='exp1')
best = fmin(fn = objective, space = space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials = trials)

