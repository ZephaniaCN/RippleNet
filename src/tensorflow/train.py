import tensorflow as tf
import numpy as np
from pathlib import Path
from src.tensorflow.model.ripple_net import RippleNet
from src.tensorflow.model.ripple_net_plus import RippleNetPlus
from src.tensorflow.data_loader import Dataset


class SummaryWriters:
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
            self.writer_dict[key] = tf.summary.FileWriter(path, sess.graph)

    def set_mode(self, mode:str):
        self.mode = mode

    def simple_value(self, tag_name, value, epoch):
        summary = tf.Summary()
        summary.value.add(tag=tag_name, simple_value=value)
        try:
            self.writer_dict[self.mode].add_summary(value, epoch)
        except:
            raise NotImplementedError('writer not set sess, use set_session')
class EpochTrainer:
    def __init__(self,sess, model, dataset:Dataset, summary_writers:SummaryWriters,batch_size):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.session = sess
        self.summary_writer = summary_writers.set_session(sess)
    def train(self,show_loss=True,show_eval=True):
        self.dataset.set_mode('train')
        self.summary_writer.set_mode('train')
        dataloader = self.dataset.data_loader(self.batch_size, self.model)
        for feed_dict in dataloader:
            _,loss = self.model.train(self.session, feed_dict)
            if show_loss:
                self.summary_writer.simple_value('loss',loss)
            if show_eval:
                self.eval('train')
    def eval(self,model, show_eval=True):
        self.dataset.set_mode(model)
        self.summary_writer.set_mode(model)
        dataloader = self.dataset.data_loader(self.batch_size, self.model)

        auc_list = []
        acc_list = []
        for feed_dict in dataloader:
            batch_auc,batch_acc = self.model.train(self.session, feed_dict)
            auc_list.append(batch_auc)
            acc_list.append(batch_acc)

        auc,acc = float(np.mean(auc_list)), float(np.mean(acc_list))
        if show_eval:
            self.summary_writer.simple_value('auc',auc)
            self.summary_writer.simple_value('acc',acc)
        return auc, acc






def step_train():
    def train(args, data_info, show_loss):
        train_data = data_info[0]
        eval_data = data_info[1]
        # test_data = data_info[2]
        n_entity = data_info[3]
        n_relation = data_info[4]
        ripple_set = data_info[5]



        model = RippleNetPlus(args, n_entity, n_relation) if args.model == 'model' else RippleNet(args,
                                                                                                            n_entity,
                                                                                                            n_relation)
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            for step in range(args.n_epoch):
                # training

                start = 0
                while start < train_data.shape[0]:
                    _, loss = model.train(
                        sess, get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
                    # print((step + start / train_data.shape[0])*1000)
                    # model_writer.add_summary(o_vals, (step+start / train_data.shape[0])*1000)
                    start += args.batch_size
                eval_auc, eval_acc = evaluation(sess, args, model, eval_data, ripple_set, args.batch_size)

            return eval_acc



def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    #test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    dp={'dim':args.dim,
        'n_hop':args.n_hop,
        'l2_weight':args.l2_weight,
        'kg_weight':args.kge_weight,
        'lr':args.lr,
        'n_memory':args.n_memory,
        'embed_size':args.embed_size
    }

    model = RippleNetPlus(args, n_entity, n_relation) if args.model=='model' else RippleNet(args,n_entity,n_relation)
    #parameters_summary =tf.summary.merge_all()
    #gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epoch):
                # training

            start = 0
            while start < train_data.shape[0]:
                _, loss = model.train(
                    sess, get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
                #print((step + start / train_data.shape[0])*1000)
                #model_writer.add_summary(o_vals, (step+start / train_data.shape[0])*1000)
                loss_summary = tf.Summary()
                loss_summary.value.add(tag='loss', simple_value=loss)
                train_writer.add_summary(loss_summary,(step+start / train_data.shape[0])*1000)
                start += args.batch_size
            # evaluation
            # train_auc, train_acc = evaluation(sess, args, model, train_data, ripple_set, args.batch_size)
            # train_summary = tf.Summary()
            # train_summary.value.add(tag='auc', simple_value=train_auc)
            # train_summary.value.add(tag='acc', simple_value=train_acc)
            # train_writer.add_summary(train_summary, step)
            eval_auc, eval_acc = evaluation(sess, args, model, eval_data, ripple_set, args.batch_size)
            eval_summary = tf.Summary()
            eval_summary.value.add(tag='auc', simple_value=eval_auc)
            eval_summary.value.add(tag='acc', simple_value=eval_acc)
            eval_writer.add_summary(eval_summary, step)
            # test_auc, test_acc = evaluation(sess, args, model, test_data, ripple_set, args.batch_size)
            # test_summary = tf.Summary()
            # test_summary.value.add(tag='auc', simple_value=test_auc)
            # test_summary.value.add(tag='acc', simple_value=test_acc)
            # test_writer.add_summary(test_summary, step)
            #print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
            #      % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))
        return eval_acc





def evaluation(sess, args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    while start < data.shape[0]:
        auc, acc = model.eval(sess, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list))
