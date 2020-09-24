import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F 

import numpy as np

from Utils.data_utils import *

#################################################################################################################
# For training
#################################################################################################################

# Dataset for implicit feedback
class implicit_CF_dataset(data.Dataset):

    def __init__(self, user_count, item_count, rating_mat, interactions, num_ns):
        """
        Parameters
        ----------
        user_count : int
            num. users
        item_count : int
            num. items
        rating_mat : dict
            user-item rating matrix
        interactions : list
            total train interactions, each instance has a form of (user, item, 1)
        num_ns : int
            num. negative samples
        is_train : bool
        """
        super(implicit_CF_dataset, self).__init__()
        
        self.user_count = user_count
        self.item_count = item_count
        self.rating_mat = rating_mat
        self.num_ns = num_ns
        self.interactions = interactions
        self.is_train = is_train
        

    def negative_sampling(self):
        """conduct the negative sampling
        """
        
        self.train_arr = []
        sample_list = np.random.choice(list(range(self.item_count)), size = 10 * len(self.interactions) * self.num_ns)
        
        sample_idx = 0
        for user, pos_item, _ in self.interactions:
            ns_count = 0
            
            while True:
                neg_item = sample_list[sample_idx]
                if not is_visited(self.rating_mat, user, neg_item):
                    self.train_arr.append((user, pos_item, neg_item))
                    sample_idx += 1
                    ns_count += 1
                    if ns_count == self.num_ns:
                        break
                        
                sample_idx += 1
    

    def __len__(self):
        return len(self.interactions) * self.num_ns
        

    def __getitem__(self, idx):
        return self.train_arr[idx][0], self.train_arr[idx][1], self.train_arr[idx][2]


class implicit_CF_dataset_AE(data.Dataset):
    def __init__(self, user_count, item_count, rating_mat):
        super(implicit_CF_dataset_AE, self).__init__()
        
        self.user_count = user_count
        self.item_count = item_count
        
        self.R = torch.zeros((user_count, item_count))
        for user in rating_mat:
            items = list(rating_mat[user].keys())
            self.R[user][items] = 1.
        

    def __len__(self):
        return self.user_count
        

    def __getitem__(self, idx):
        return idx, self.R[idx]


#################################################################################################################
# For test
#################################################################################################################

class implicit_CF_dataset_test(data.Dataset):
    """
        Test Dataset for Leave-One-Out evaluation protocol.
        It is used for a large model which cannot compute the total rating matrix at once.
    """
    def __init__(self, user_count, test_sample, valid_sample, candidates, batch_size=1024):
        """
        Parameters
        ----------
        user_count : int
            num. users
        test_sample : dict
            sampled test item for each user
        valid_sample : dict
            sampled valid item for each user
        candidates : dict
            sampled candidate items for each user
        batch_size : int, optional
            by default 1024
        """
        super(implicit_CF_dataset_test, self).__init__()

        self.test_item =[]  
        self.valid_item = [] 
        self.candidates = []

        num_candidates = len(candidates[0])

        for user in range(user_count):
            if user not in test_sample:
                self.test_item.append([0])
                self.valid_item.append([0])
                self.candidates.append([0] * num_candidates)
            else:
                self.test_item.append([int(test_sample[user])])
                self.valid_item.append([int(valid_sample[user])])
                self.candidates.append(list(candidates[user].keys()))

        self.test_item = torch.LongTensor(self.test_item)
        self.valid_item = torch.LongTensor(self.valid_item)
        self.candidates = torch.LongTensor(self.candidates)

        self.user_list = torch.LongTensor(list(test_sample.keys()))

        self.batch_start = 0
        self.batch_size = batch_size


    def __len__(self):
        return len(self.test_item)


    def get_next_batch_users(self):
        """get the next batch of test sers

        Returns
        -------
        self.user_list[batch_start: batch_end] : 1-D LongTensor
            next batch of users
        
        is_last_batch : bool
        """
        batch_start = self.batch_start
        batch_end = self.batch_start + self.batch_size

        # if it is the last batch
        if batch_end >= len(self.user_list):
            batch_end = len(self.user_list)
            self.batch_start = 0
            is_last_batch = True
        else:
            self.batch_start += self.batch_size
            is_last_batch = False

        return self.user_list[batch_start: batch_end], is_last_batch


    def get_next_batch(self, batch_user):
        """get next test batch (i.e., test samples, valid samples, candidates)

        Parameters
        ----------
        batch_user : 1-D LongTensor
            current batch of test users

        Returns
        -------
        batch_test_items: 2-D LongTensor (batch_size x 1)
        batch_valid_items: 2-D LongTensor (batch_size x 1)
        batch_candidates: 2-D LongTensor (batch_size x num. candidates)
        """
        batch_test_items = torch.index_select(self.test_item, 0, batch_user)
        batch_valid_items = torch.index_select(self.valid_item, 0, batch_user)
        batch_candidates = torch.index_select(self.candidates, 0, batch_user)

        return batch_test_items, batch_valid_items, batch_candidates

  

class implicit_CF_dataset_AE_test(implicit_CF_dataset_test):
    def __init__(self, user_count, item_count, rating_mat, test_sample, valid_sample, candidates, batch_size=128):

        implicit_CF_dataset_test.__init__(user_count, test_sample, valid_sample, candidates, batch_size)
        
        self.user_count = user_count
        self.item_count = item_count
        self.batch_size = batch_size
        
        self.R = torch.zeros((user_count, item_count))
        for user in rating_mat:
            items = list(rating_mat[user].keys())
            self.R[user][items] = 1.
        

    def get_next_batch_users(self):
        """get the next batch of test sers

        Returns
        -------
        self.user_list[batch_start: batch_end] : 1-D LongTensor
            next batch of users
        
        is_last_batch : bool
        """
        batch_start = self.batch_start
        batch_end = self.batch_start + self.batch_size

        # if it is the last batch
        if batch_end >= len(self.user_list):
            batch_end = len(self.user_list)
            self.batch_start = 0
            is_last_batch = True
        else:
            self.batch_start += self.batch_size
            is_last_batch = False

        return self.user_list[batch_start: batch_end], self.R[batch_start: batch_end], is_last_batch