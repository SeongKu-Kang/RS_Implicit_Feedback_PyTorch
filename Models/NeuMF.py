import torch.nn.functional as F
import torch.nn as nn
import torch

class NeuMF(nn.Module):
	def __init__(self, user_count, item_count, dim, num_hidden_layer, gpu):
		"""
		Parameters
		----------
		user_count : int
		item_count : int
		dim : int
			embedding dimension
		num_hidden_layer : int
			num. of hidden layers in MLP
		gpu : if available
		"""
		super(NeuMF, self).__init__()

		self.user_count = user_count
		self.item_count = item_count

		self.user_list = torch.LongTensor([i for i in range(user_count)])
		self.item_list = torch.LongTensor([i for i in range(item_count)])

		if gpu != None:
			self.user_list = self.user_list.to(gpu)
			self.item_list = self.item_list.to(gpu)

		
		# User / Item Embedding
		self.user_emb_MF = nn.Embedding(self.user_count, dim)
		self.item_emb_MF = nn.Embedding(self.item_count, dim)

		self.user_emb_MLP = nn.Embedding(self.user_count, dim)
		self.item_emb_MLP = nn.Embedding(self.item_count, dim)

		self.sim_type = 'network'

		nn.init.normal_(self.user_emb_MF.weight, mean=0., std= 0.01)
		nn.init.normal_(self.item_emb_MF.weight, mean=0., std= 0.01)
		
		nn.init.normal_(self.user_emb_MLP.weight, mean=0., std= 0.01)
		nn.init.normal_(self.item_emb_MLP.weight, mean=0., std= 0.01)

		# Layer configuration
		##  MLP Layers
		MLP_layers = []
		layers_shape = [dim * 2]
		for i in range(num_hidden_layer):
			layers_shape.append(layers_shape[-1] // 2)
			MLP_layers.append(nn.Linear(layers_shape[-2], layers_shape[-1]))
			MLP_layers.append(nn.ReLU())

		print("MLP Layer Shape ::", layers_shape)
		self.MLP_layers = nn.Sequential(* MLP_layers)
		
		## Final Layer
		self.final_layer  = nn.Linear(layers_shape[-1] * 2, 1)
		
		self._init_weights()

		# Loss function
		self.BCE_loss = nn.BCEWithLogitsLoss(reduction='sum')


	def _init_weights(self):
		# Layer initialization
		for m in self.MLP_layers:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
		nn.init.kaiming_uniform_(self.final_layer.weight, a=1, nonlinearity='relu')

		for m in self.modules():
			if isinstance(m, nn.Linear) and m.bias is not None:
				m.bias.data.zero_()


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

		pos_score = self.forward_no_neg(batch_user, batch_pos_item)	 # bs x 1
		neg_score = self.forward_no_neg(batch_user, batch_neg_item)	 # bs x 1

		output = (pos_score, neg_score)

		return output


	def forward_no_neg(self, batch_user, batch_item):
		"""forward without negative items

		Parameters
		----------
		batch_user : 1-D LongTensor (batch_size)
		batch_item : 1-D LongTensor (batch_size)

		Returns
		-------
		output : 2-D LongTensor (batch_size x 1)
		"""
		
		# MF
		u_mf = self.user_emb_MF(batch_user)			# batch_size x dim
		i_mf = self.item_emb_MF(batch_item)			# batch_size x dim
		
		mf_vector = (u_mf * i_mf)					# batch_size x dim

		# MLP
		u_mlp = self.user_emb_MLP(batch_user)		# batch_size x dim
		i_mlp = self.item_emb_MLP(batch_item)		# batch_size x dim

		mlp_vector = torch.cat([u_mlp, i_mlp], dim=-1)
		mlp_vector = self.MLP_layers(mlp_vector)

		predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
		output = self.final_layer(predict_vector) 

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
		pos_score, neg_score = output[0], output[1]

		pred = torch.cat([pos_score, neg_score], dim=0)
		gt = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
		
		return self.BCE_loss(pred, gt)


	def forward_multi_items(self, batch_user, batch_items):
		"""forward when we have multiple items for a user,
			Usually for evaluation purpose

		Parameters
		----------
		batch_user : 1-D LongTensor (batch_size)
		batch_items : 2-D LongTensor (batch_size x k)

		Returns
		-------
		score : 2-D FloatTensor (batch_size x k)
		"""
		batch_user = batch_user.unsqueeze(-1)
		batch_user = torch.cat(batch_items.size(1) * [batch_user], 1)

		score = self.forward_no_neg(batch_user, batch_items).squeeze(-1)
			
		return score

