from pathlib import Path
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
    'file_name':'origin_ripple_net',
    'n_epoch':10,
    'batch_size':1024,
    'show_eval':True,
    'show_loss':True,
    'test':True,
    'save_model':True,
    'show_train_eval':True
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
    'show_train_eval': True
}
# ripple net plus default args
exp2_args = {
    'model':'ripple_net_plus',
    'dataset_path': Path('../../data'),
    'dataset':'movie',
    'dim':16,
    'n_hop':3,
    'kge_weight':0.01,
    'l2_weight':1e-7,
    'lr':0.02,
    'n_memory':32,
    'log_path':Path('../../logs'),
    'model_path':Path('../../models'),
    'file_name':'ripple_net_plus',
    'n_epoch':20,
    'batch_size':1024,
    'show_eval':True,
    'show_loss':True,
    'test':False,
    'save_model':True,
    'show_train_eval':True
}

def args_convert(args):
    target_args = dict()
    target_args['static_args']={
        'model_args': {
           'dim': args['dim'],
           'n_hop': args['n_hop'],
           'kge_weight': args['kge_weight'],
           'l2_weight': args['l2_weight'],
           'lr': args['lr'],
           'n_memory': args['n_memory'],
       },
       'dataset_args': {
           'root_path': args['dataset_path'],
           'dataset': args['dataset'],
           'dim': args['dim'],
           'n_hop': args['n_hop'],
           'n_memory': args['n_memory']
       },
        'model':args['model'],
        'log_path': args['log_path'],
        'model_path': args['model_path'],
        'file_name': args['file_name'],
        'n_epoch': args['n_epoch'],
        'batch_size': args['batch_size']
    }
    target_args['runtime_args']={
        'show_eval': args['show_eval'],
        'show_loss': args['show_loss'],
        'test': args['test'],
        'save_model': args['save_model'],
        'show_train_eval': args['show_train_eval']
    }
    return target_args