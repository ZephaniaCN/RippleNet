
import numpy as np
import hyperopt_model
from hyperopt import hp,fmin,tpe,Trials
from hyperopt.mongoexp import MongoTrials



np.random.seed(555)

MAX_EVALS=500

space = {
    'n_hop': hp.quniform('n_hop', 1, 4, 1),
    'dim':hp.quniform('dim',2,128,1),
    'kge_weight': hp.loguniform('kge_weight',np.log(1e-9),np.log(10)),
    'l2_weight': hp.loguniform('l2_weight',np.log(1e-9),np.log(10)),
    'n_memory':hp.quniform('n_memory',2,128,1),
    'embed_size':hp.quniform('embed_size',2,128,1),
    'lr':hp.loguniform('lr',np.log(1e-5),np.log(1))
}

trials=Trials()
#trials = MongoTrials('mongo://localhost:27017/mnn_db/jobs', exp_key='exp1')
best = fmin(fn = hyperopt_model.objective, space = space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials = trials)

print(trials.best_trial)
