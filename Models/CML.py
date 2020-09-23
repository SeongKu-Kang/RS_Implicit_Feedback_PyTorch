import torch
import torch.nn as nn
import torch.nn.functional as F

class CML(nn.Module):
	def __init__(self, user_count, item_count, dim, margin, gpu):
		"""
		Parameters
		----------
		user_count : int
		item_count : int
		dim : int
		margin : float
		"""
		super(CML, self).__init__()
		self.user_count = user_count
		self.item_count = item_count

		self.user_list = torch.LongTensor([i for i in range(user_count)])
		self.item_list = torch.LongTensor([i for i in range(item_count)])

		if gpu != None:
			self.user_list = self.user_list.to(gpu)
			self.item_list = self.item_list.to(gpu)
			
		# User / Item Embedding
		self.user_emb = nn.Embedding(self.user_count, dim, max_norm=1.)
		self.item_emb = nn.Embedding(self.item_count, dim, max_norm=1.)

		nn.init.normal_(self.user_emb.weight, mean=0., std= 1 / (dim ** 0.5))
		nn.init.normal_(self.item_emb.weight, mean=0., std= 1 / (dim ** 0.5))

		self.margin = margin

		# user-item similarity type
		self.sim_type = 'L2 dist'


	def forward(self, batch_user, batch_pos_item, batch_neg_item):
		"""
		Parameters
		----------
		batch_user : 1-D LongTensor (batch_size)
		batch_pos_item : 1-D LongTensor (batch_size)
		batch_neg_item : 1-D LongTensor (batch_size)

		Returns
		-------
		output : 
			Model output to calculate its loss function
		"""
		
		u = self.user_emb(batch_user)			
		i = self.item_emb(batch_pos_item)		
		j = self.item_emb(batch_neg_item)	
		
		pos_dist = ((u - i) ** 2).sum(dim=1, keepdim=True)
		neg_dist = ((u - j) ** 2).sum(dim=1, keepdim=True)

		output = (pos_dist, neg_dist)

		return output


	def get_loss(self, output):
		"""Compute the loss function with the model output

		Parameters
		----------
		output : 
			model output (results of forward function)

		Returns
		-------
		loss : float
		"""
		pos_dist, neg_dist = output[0], output[1]
		loss = F.relu(self.margin + pos_dist - neg_dist).sum()
		
		return loss


	def forward_multi_items(self, batch_user, batch_items):
		"""forward when we have multiple items for a user,
			Usually for evaluation purpose

		Parameters
		----------
		batch_user : 1-D LongTensor (batch_size)
		batch_items : 2-D LongTensor (batch_size x k)

		Returns
		-------
		dist : 2-D FloatTensor (batch_size x k)
		"""
		batch_user = batch_user.unsqueeze(-1)
		batch_user = torch.cat(batch_items.size(1) * [batch_user], 1)
			
		u = self.user_emb(batch_user)		# batch_size x k x dim
		i = self.item_emb(batch_items)		# batch_size x k x dim
		
		dist = ((u - i) ** 2).sum(dim=-1, keepdim=False)
		
		return dist


	def get_embedding(self):
		"""get total embedding of users and items

		Returns
		-------
		users : 2-D FloatTensor (num. users x dim)
		items : 2-D FloatTensor (num. items x dim)
		"""
		users = self.user_emb(self.user_list)
		items = self.item_emb(self.item_list)

		return users, items



