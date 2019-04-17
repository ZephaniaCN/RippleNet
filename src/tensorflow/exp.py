from src.tensorflow.train import  run_exp
import src.tensorflow.args as args
import numpy as np
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#np.random.seed(555)

run_exp(args.args_convert(args.ripple_net_plus_book_args))