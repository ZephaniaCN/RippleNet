import collections
import os
import numpy as np
import logging
from pathlib import Path
logger = logging.getLogger()


class Dataset:
    def __init__(self, root_path,dataset,n_hop,n_memory,dim, mode='train'):
        # data load
        self.data_path = Path(root_path,dataset)

        self.eval_ratio = 0.2
        self.test_ratio = 0.2

        self.n_hop = n_hop
        # size of each hop
        self.n_memory = n_memory
        # dims of each embeding
        self.dim =dim
        #[(user_0,item_0,label_0),(user_1,item_1,label_1)...]
        self.train_data, self.eval_data, self.test_data, self.user_history_dict = self.load_rating()
        self.kg, self.n_relation, self.n_entity = self.load_kg()
        # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
        self.ripple_set = self.get_ripple_set(self.n_hop,n_memory, self.kg, self.user_history_dict)

        # switch train/eval/test
        self.dataset_switch = {
            'train': self.train_data,
            'eval': self.eval_data,
            'test': self.test_data,
        }

        self.data = self.dataset_switch[mode]

    def get_n_enitity_relation(self):
        return self.n_entity,self.n_relation

    def set_mode(self,mode):
        self.data = self.dataset_switch[mode]
    def data_loader(self,batch_size,model,shuffle=True):
        end = 0
        if shuffle:
            np.random.shuffle(self.data)
        while end<=self.data.shape[0]:
            start = end
            end += batch_size
            feed_dict = dict()
            feed_dict[model.items] = self.data[start:end, 1]
            feed_dict[model.labels] = self.data[start:end, 2]
            for i in range(self.n_hop):
                feed_dict[model.memories_h[i]] = [self.ripple_set[user][i][0] for user in self.data[start:end, 0]]
                feed_dict[model.memories_r[i]] = [self.ripple_set[user][i][1] for user in self.data[start:end, 0]]
                feed_dict[model.memories_t[i]] = [self.ripple_set[user][i][2] for user in self.data[start:end, 0]]
            yield feed_dict

    def load_rating(self):
        logging.info('reading rating file ...')

        # reading ratings
        ratings = self.load_data('ratings_final')

        if logger.getEffectiveLevel() == logging.DEBUG:
            n_user = len(set(ratings[:, 0]))
            n_item = len(set(ratings[:, 1]))
            logging.debug('users num:{}\n items num: {}'.format(n_user,n_item))
        return self.dataset_split(ratings)
    def dataset_split(self, ratings):
        logger.info('splitting dataset ...')

        # train:eval:test = 6:2:2
        n_ratings = ratings.shape[0]

        eval_indices = np.random.choice(n_ratings, size=int(n_ratings * self.eval_ratio), replace=False)
        left = set(range(n_ratings)) - set(eval_indices)
        test_indices = np.random.choice(list(left), size=int(n_ratings * self.test_ratio), replace=False)
        train_indices = list(left - set(test_indices))
        logging.debug('rating lens:{}\ntrain lens:{}\neval lens:{}\ntest lens{}'.format\
                          (n_ratings,len(train_indices), len(eval_indices), len(test_indices)))

        # traverse training data, only keeping the users with positive ratings
        user_history_dict = dict()
        for i in train_indices:
            user = ratings[i][0]
            item = ratings[i][1]
            rating = ratings[i][2]
            if rating == 1:
                if user not in user_history_dict:
                    user_history_dict[user] = []
                user_history_dict[user].append(item)

        train_indices = [i for i in train_indices if ratings[i][0] in user_history_dict]
        eval_indices = [i for i in eval_indices if ratings[i][0] in user_history_dict]
        test_indices = [i for i in test_indices if ratings[i][0] in user_history_dict]

        logging.debug('after clean\nrating lens:{}\ntrain lens:{}\neval lens:{}\ntest lens{}'.format \
                          (n_ratings, len(train_indices), len(eval_indices), len(test_indices)))

        train_data = ratings[train_indices]
        eval_data = ratings[eval_indices]
        test_data = ratings[test_indices]

        return train_data, eval_data, test_data, user_history_dict
    def load_data(self,filename):
        loading_file = self.data_path/(filename+'.np')
        if loading_file.exists():
            loaded_file = np.load(loading_file)
        else:
            rating_file_txt = self.data_path/(filename+'.txt')
            loaded_file = np.loadtxt(rating_file_txt, dtype=np.int32)
            np.save(loading_file, loaded_file)
        return loaded_file
    def load_kg(self):
        logger.info('reading KG file ...')

        # reading kg file
        kg_np = self.load_data('kg_final')

        n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
        n_relation = len(set(kg_np[:, 1]))

        logging.debug('entity num:{}\n relation num: {}'.format(n_entity, n_relation))

        kg = self.construct_kg(kg_np)

        return  kg,n_relation,n_entity
    def construct_kg(self,kg_np):
        logger.info('constructing knowledge graph ...')
        kg = collections.defaultdict(list)
        for head, relation, tail in kg_np:
            kg[head].append((tail, relation))
        return kg
    def get_ripple_set(self,n_hop,n_memory, kg, user_history_dict):
        print('constructing ripple set ...')

        # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
        ripple_set = collections.defaultdict(list)

        for user in user_history_dict:
            for h in range(n_hop):
                memories_h = []
                memories_r = []
                memories_t = []

                if h == 0:
                    tails_of_last_hop = user_history_dict[user]
                else:
                    tails_of_last_hop = ripple_set[user][-1][2]

                for entity in tails_of_last_hop:
                    for tail_and_relation in kg[entity]:
                        memories_h.append(entity)
                        memories_r.append(tail_and_relation[1])
                        memories_t.append(tail_and_relation[0])

                # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
                # this won't happen for h = 0, because only the items that appear in the KG have been selected
                # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
                if len(memories_h) == 0:
                    ripple_set[user].append(ripple_set[user][-1])
                else:
                    # sample a fixed-size 1-hop memory for each user
                    replace = len(memories_h) < n_memory
                    indices = np.random.choice(len(memories_h), size=n_memory, replace=replace)
                    memories_h = [memories_h[i] for i in indices]
                    memories_r = [memories_r[i] for i in indices]
                    memories_t = [memories_t[i] for i in indices]
                    ripple_set[user].append((memories_h, memories_r, memories_t))

        return ripple_set



