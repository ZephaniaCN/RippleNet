import tensorflow as tf
import numpy as np
from pathlib import Path
from src.tensorflow.model.ripple_net import RippleNet
from src.tensorflow.model.ripple_net_plus import RippleNetPlus
from src.tensorflow.data_loader import Dataset
from logging import getLogger
logger = getLogger()

class summary_writers:
    def __init__(self,log_path:Path, file_name:Path):
        self.log_path = log_path
        self.file_name = file_name
        self.writer_dict = {
            'test':None,
            'train':None,
            'eval':None
        }
        self.mode = None

    def set_session(self, sess):
        for key in self.writer_dict.keys():
            path = self.log_path / key / self.file_name
            self.writer_dict[key] = tf.summary.FileWriter(str(path), sess.graph)

    def set_mode(self, mode:str):
        self.mode = mode

    def simple_value(self, tag_name, value, epoch):
        summary = tf.Summary()
        summary.value.add(tag=tag_name, simple_value=value)
        try:
            self.writer_dict[self.mode].add_summary(summary, epoch)
        except:
            raise NotImplementedError('writer not set sess, use set_session')
class EpochTrainer:
    def __init__(self, model, dataset:Dataset, batch_size):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size

    def train(self,  sess):
        self.dataset.set_mode('train')
        dataloader = self.dataset.data_loader(self.batch_size, self.model)
        for feed_dict in dataloader:
            _, loss = self.model.train(sess, feed_dict)
        return loss

    def eval(self,sess, mode):
        self.dataset.set_mode(mode)
        dataloader = self.dataset.data_loader(self.batch_size, self.model)

        auc_list = []
        acc_list = []
        for feed_dict in dataloader:
            batch_auc,batch_acc = self.model.eval(sess, feed_dict)
            auc_list.append(batch_auc)
            acc_list.append(batch_acc)

        auc,acc = float(np.mean(auc_list)), float(np.mean(acc_list))

        return auc, acc

class Experiement:
    def __init__(self,model,model_args,dataset_args,
                 log_path,model_path,file_name,n_epoch,batch_size):
        model_dict={
            'ripple_net': RippleNet,
            'ripple_net_plus':RippleNetPlus
        }
        self.n_epoch = n_epoch
        self.model_path = model_path/'{}_model.ckpt'.format(str(file_name))
        self.model_path = str(self.model_path)
        self.dataset = Dataset(**dataset_args)
        n_entity, n_relation = self.dataset.get_n_enitity_relation()
        self.model = model_dict[model](**model_args,n_entity=n_entity,n_relation=n_relation)
        self.trainer = EpochTrainer(self.model,self.dataset,batch_size)
        self.summary_writer = summary_writers(log_path,file_name)

    def run(self,save_model=False,show_loss=True,show_eval=True,show_train_eval=False,test=False):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.summary_writer.set_session(sess)
            for step in range(self.n_epoch):
                loss = self.trainer.train(sess)
                if show_loss:
                    self.summary_writer.set_mode('train')
                    self.summary_writer.simple_value('loss',loss, step)
                if show_train_eval:
                    auc,acc = self.trainer.eval(sess,'train')
                    self.summary_writer.simple_value('auc', auc, step)
                    self.summary_writer.simple_value('acc', acc, step)
                if show_eval:
                    auc,acc = self.trainer.eval(sess, 'eval')
                    self.summary_writer.set_mode('eval')
                    self.summary_writer.simple_value('auc', auc, step)
                    self.summary_writer.simple_value('acc', acc, step)
                if test:
                    auc, acc = self.trainer.eval(sess, 'test')
                    self.summary_writer.set_mode('test')
                    self.summary_writer.simple_value('auc', auc, step)
                    self.summary_writer.simple_value('acc', acc, step)
            if save_model:
                saver = tf.train.Saver()
                save_path = saver.save(sess, self.model_path)
                logger.info("Model saved in path: {}" .format(save_path))
            return auc, acc

def run_exp(static_args, runtime_args):
    exp = Experiement(**static_args)
    exp.run(**runtime_args)