import tensorflow as tf
from pathlib import Path
import os
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score
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
            path = self.log_path /self.file_name/ key
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
    def simple_values(self, values_dict:dict, epoch):
        for key,value in values_dict.items():
            self.simple_value(key, value, epoch)

    def flush(self):
        for writer in self.writer_dict.values():
            writer.flush()

def auc_cal(scores, labels):
    auc = roc_auc_score(y_true=labels, y_score=scores)
    return auc
def acc_cal(scores, labels):
    predictions = [1 if i >= 0.5 else 0 for i in scores]
    acc = np.mean(np.equal(predictions, labels))
    return acc
def recall_cal(scores, labels):
    predictions = (scores+0.5).astype(int)
    y_true = labels.astype(int)
    recall = recall_score(y_true=y_true,y_pred=predictions)
    return recall
def f1_cal(scores, labels):
    predictions = (scores + 0.5).astype(int)
    y_true = labels.astype(int)
    f1 = f1_score(y_true=y_true, y_pred=predictions)
    return f1

class EpochTrainer:
    def __init__(self, model, dataset:Dataset, batch_size, max_loss):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_loss = max_loss
        self.eval_fns = {
            'auc': auc_cal,
            'acc': acc_cal,
            'recall': recall_cal,
            'f1':f1_cal
        }

    def train(self,  sess):
        self.dataset.set_mode('train')
        dataloader = self.dataset.data_loader(self.batch_size, self.model)
        for feed_dict in dataloader:
            _, loss = self.model.train(sess, feed_dict)
            if loss>self.max_loss:
                break
        return loss

    def eval(self,sess, mode):
        self.dataset.set_mode(mode)
        dataloader = self.dataset.data_loader(self.batch_size, self.model)

        eval_list = {key:[] for key in self.eval_fns.keys()}
        for feed_dict in dataloader:
            labels, scores = self.model.eval_data(sess, feed_dict)
            for key in self.eval_fns.keys():
                eval_list[key].append(self.eval_fns[key](scores, labels))

        return {key:float(np.mean(eval_list[key])) for key in self.eval_fns.keys()}


class Experiement:
    def __init__(self,model,model_args,dataset_args,
                 log_path,model_path,file_name,n_epoch,batch_size,max_loss=20):
        model_dict={
            'ripple_net': RippleNet,
            'ripple_net_plus':RippleNetPlus
        }
        self.n_epoch = n_epoch
        if not Path(model_path).exists():
            os.mkdir(model_path)
        self.model_path = model_path/'{}_model.ckpt'.format(str(file_name))
        self.model_path = str(self.model_path)
        self.max_loss = max_loss
        self.dataset = Dataset(**dataset_args)
        n_entity, n_relation = self.dataset.get_n_enitity_relation()
        self.model = model_dict[model](**model_args,n_entity=n_entity,n_relation=n_relation)
        self.trainer = EpochTrainer(self.model,self.dataset,batch_size,max_loss)
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
                    eval_dict = self.trainer.eval(sess,'train')
                    self.summary_writer.simple_values(eval_dict,step)

                if show_eval:
                    eval_dict = self.trainer.eval(sess, 'eval')
                    self.summary_writer.set_mode('eval')
                    self.summary_writer.simple_values(eval_dict,step)

                if test:
                    eval_dict = self.trainer.eval(sess, 'test')
                    self.summary_writer.set_mode('test')
                    self.summary_writer.simple_values(eval_dict,step)

                if loss > self.max_loss:
                    break

            self.summary_writer.flush()

            if save_model:
                saver = tf.train.Saver()
                save_path = saver.save(sess, self.model_path)
                logger.info("Model saved in path: {}" .format(save_path))


            return eval_dict

def run_exp(args):
    static_args = args['static_args']
    runtime_args = args['runtime_args']
    exp = Experiement(**static_args)
    res = exp.run(**runtime_args)
    return res