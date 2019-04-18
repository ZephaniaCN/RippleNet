from pathlib import Path

ripple_net_plus_movie_args = {
    'model':'ripple_net_plus',
    'dataset_path': Path('../../data'),
    'dataset':'movie',
    'dim':16,
    'n_hop':3,
    'kge_weight':0.01,
    'l2_weight':1e-7,
    'lr':0.02,
    'n_memory':32,
    'dropout':0.8,
    'n_epoch':15,
    'batch_size':1024,
    'use_hyperopt': False,
    'test':True,
    'eval':True
}
#run_ripple_net_plus_book_6.9e-05lr_4.2e-02kg_4.8e-08l2_0.50dp_44m_2h_44d_basic
ripple_net_plus_book_args = {
    'model':'ripple_net_plus',
    'dataset_path': Path('../../data'),
    'dataset':'book',
    'dim':4,
    'n_hop':2,
    'kge_weight':0.01,
    'l2_weight':1e-5,
    'lr':0.001,
    'n_memory':32,
    'dropout':0.2,
    'n_epoch':15,
    'batch_size':1024,
    'reporter_mode':'print',
    'use_hyperopt': False, # 是否使用nni自动调参
    'hyper_key': 'auc',
    'test':True,
    'eval':True,
    'eval_train':True
}

ripple_net_plus_hyper_book_args = {
    'model':'ripple_net_plus',
    'dataset_path': Path('../../data'),
    'dataset':'book',
    'dim':4,
    'n_hop':2,
    'kge_weight':0.01,
    'l2_weight':1e-5,
    'lr':0.001,
    'n_memory':32,
    'dropout':0.2,
    'n_epoch':15,
    'batch_size':1024,
    'reporter_mode':'tensorboard',
    'use_hyperopt': True, # 是否使用nni自动调参
    'hyper_key': 'auc',
    'test':False,
    'eval':True,
    'eval_train':False
}
def args_convert(args):
    target_args = {'model_args': {
        'dim': args['dim'],
        'n_hop': args['n_hop'],
        'kge_weight': args['kge_weight'],
        'l2_weight': args['l2_weight'],
        'dropout': args['dropout']
    }, 'dataset_args': {
        'root_path': args['dataset_path'],
        'dataset': args['dataset'],
        'dim': args['dim'],
        'n_hop': args['n_hop'],
        'n_memory': args['n_memory']
    },
        'model': args['model'],
        #'file_name': args['file_name'],
        'n_epoch': args['n_epoch'],
        'lr': args['lr'],
        'batch_size': args['batch_size'],
        'eval':args['eval'],
        'test':args['test'],
        'eval_train':args['eval_train'],
        'use_hyperopt':args['use_hyperopt'],
        'hyper_key':args['hyper_key'],
        'reporter_mode':args['reporter_mode']
        }
    return target_args