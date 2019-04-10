# import sys
# sys.path.append('../..')
from src.pytorch.args import ripple_net_plus_book_args, args_convert
from src.pytorch.experiment import Trainer
from src.pytorch.model.ripple_net_plus import AggregateFnc
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#
trainer=Trainer(**args_convert(ripple_net_plus_book_args))
trainer.train()
