from pathlib import Path
# tensoroverflow
# ripple net movie default args
ripple_net_movie_args = {
    'model':'ripple_net',
    'dataset_path': Path('../../data'),
    'dataset':'movie',
    'dim':16,
    'n_hop':2,
    'kge_weight':0.01,
    'l2_weight':1e-7,
    'lr':0.02,
    'n_memory':32,
    'log_path':Path('../../logs'),
    'model_path':Path('../../models'),
    'file_name':'ripple_net_movie',
    'n_epoch':10,
    'batch_size':1024,
    'show_eval':True,
    'show_loss':True,
    'test':True,
    'save_model':True,
    'show_train_eval':True,
    'load':True,
    'max_loss':100
}
ripple_net_book_args = {
    'model': 'ripple_net',
    'dataset_path': Path('../../data'),
    'dataset': 'book',
    'dim': 4,
    'n_hop': 2,
    'kge_weight': 1e-2,
    'l2_weight': 1e-5,
    'lr': 1e-3,
    'n_memory': 32,
    'log_path': Path('../../logs'),
    'model_path': Path('../../models'),
    'file_name': 'ripple_net_book',
    'n_epoch': 10,
    'batch_size': 1024,
    'show_eval': True,
    'show_loss': True,
    'test': True,
    'save_model': True,
    'show_train_eval': True,
    'load':False,
    'max_loss': 100

}
# ripple net plus default args
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
    'predict_mode':'dense',
    'log_path':Path('../../logs'),
    'model_path':Path('../../models'),
    'file_name':'ripple_net_plus_movie',
    'n_epoch':20,
    'batch_size':1024,
    'eval':True,
    'test':True,
}
#run_ripple_net_plus_book_6.9e-05lr_4.2e-02kg_4.8e-08l2_0.50dp_44m_2h_44d_basic
ripple_net_plus_book_args = {
    'model':'ripple_net_plus',
    'dataset_path': Path('../../data'),
    'dataset':'book',
    'dim':6,
    'n_hop':3,
    'kge_weight':0.01,
    'l2_weight':1e-5,
    'lr':0.001,
    'n_memory':32,
    'dropout':0.8,
    'n_epoch':15,
    'batch_size':1024,
    'use_hyperopt': False,
    'test':True,
    'eval':True
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
        'use_hyperopt':args['use_hyperopt']
        }
    return target_args