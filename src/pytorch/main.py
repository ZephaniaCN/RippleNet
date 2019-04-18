import logging
logger = logging.getLogger()
import sys
sys.path.append('../..')
import src.pytorch.args as args
from src.pytorch.experiment import run_exp



if __name__ == '__main__':
    try:
        run_exp(args.ripple_net_plus_hyper_book_args)
    except Exception as exception:
        logger.exception(exception)
        raise
