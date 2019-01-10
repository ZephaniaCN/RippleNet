import tensorflow as tf
import numpy as np
from src.model import RippleNet
from src.ripple_net_plus.model import RippleNetPlus



def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    PFILE = '{}_hops'.format(args.n_hop)

    model = RippleNetPlus(args, n_entity, n_relation) if args.model=='ripple_net_plus' else RippleNet(args,n_entity,n_relation)
    #parameters_summary =tf.summary.merge_all()
    with tf.Session() as sess:

        test_writer = tf.summary.FileWriter('./logs/{}/{}_test'.format(PFILE,args.model),sess.graph)
        train_writer = tf.summary.FileWriter('./logs/{}/{}_train'.format(PFILE,args.model), sess.graph)
        eval_writer = tf.summary.FileWriter('./logs/{}/{}_eval'.format(PFILE,args.model), sess.graph)
        model_writer = tf.summary.FileWriter('./logs/{}/{}_parameters'.format(PFILE,args.model), sess.graph)
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epoch):
                # training
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]:
                _, loss = model.train(
                    sess, get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
                print((step + start / train_data.shape[0])*1000)
                #model_writer.add_summary(o_vals, (step+start / train_data.shape[0])*1000)
                loss_summary = tf.Summary()
                loss_summary.value.add(tag='loss', simple_value=loss)
                train_writer.add_summary(loss_summary,(step+start / train_data.shape[0])*1000)
                start += args.batch_size


            # evaluation
            train_auc, train_acc = evaluation(sess, args, model, train_data, ripple_set, args.batch_size)
            train_summary = tf.Summary()
            train_summary.value.add(tag='auc', simple_value=train_auc)
            train_summary.value.add(tag='acc', simple_value=train_acc)
            train_writer.add_summary(train_summary, step)
            eval_auc, eval_acc = evaluation(sess, args, model, eval_data, ripple_set, args.batch_size)
            eval_summary = tf.Summary()
            eval_summary.value.add(tag='auc', simple_value=eval_auc)
            eval_summary.value.add(tag='acc', simple_value=eval_acc)
            eval_writer.add_summary(eval_summary, step)
            test_auc, test_acc = evaluation(sess, args, model, test_data, ripple_set, args.batch_size)
            test_summary = tf.Summary()
            test_summary.value.add(tag='auc', simple_value=test_auc)
            test_summary.value.add(tag='acc', simple_value=test_acc)
            test_writer.add_summary(test_summary, step)
            print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                  % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))


def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    for i in range(args.n_hop):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
    return feed_dict


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
