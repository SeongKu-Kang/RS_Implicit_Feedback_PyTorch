import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bp

class CDAE(nn.Module):
	def __init__(self, user_count, item_count, hidden_dim, noise_level, num_ns, gpu):
		"""
		Parameters
		----------
		user_count : int
		item_count : int
		hidden_dim : int
			bottleneck layer dimension
		noise_level : float
			noise level
		gpu : if available
		"""
		super(CDAE, self).__init__()
		self.user_count = user_count
		self.item_count = item_count

		self.user_list = torch.LongTensor([i for i in range(user_count)]).to(gpu)

		self.hidden_dim = hidden_dim
		self.num_ns = num_ns

		# user embedding
		self.user_emb = nn.Embedding(self.user_count, self.hidden_dim)
		nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)

		# network
		self.Encoder = nn.Linear(self.item_count, self.hidden_dim)
		self.Decoder = nn.Linear(self.hidden_dim, self.item_count)

		# functions for training
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()

		self.noise_level = noise_level
		self.drop_out = nn.Dropout(p=noise_level)

		self.gpu = gpu
		self.loss = torch.nn.BCEWithLogitsLoss(reduction='sum')

		# user-item similarity type
		self.sim_type = 'AE'

		

	def forward(self, batch_user, batch_user_R):

		batch_user_R_corrupted = self.drop_out(batch_user_R) 
		
		hidden_states = self.Encoder(batch_user_R_corrupted) + self.user_emb(batch_user)
		hidden_states = self.relu(hidden_states)
		output = self.Decoder(hidden_states)
		
		return output


	def get_loss(self, output, batch_user_R):

		with torch.no_grad():

			batch_num_ns_max = int(batch_user_R.sum(1).max() * self.num_ns)
			batch_sampled_max = torch.multinomial((1 - batch_user_R), batch_num_ns_max, replacement=True)

			negative_samples = []
			users = []
			for u in range(batch_user_R.size(0)):
				# num_n = max(int(batch_user_R[u].sum() * self.num_ns), self.num_ns)
				num_ns_per_user = int(batch_user_R[u].sum() * self.num_ns)
				negative_samples.append(batch_sampled_max[u][: num_ns_per_user])
				users.extend([u] * num_ns_per_user)

			negative_samples = torch.cat(negative_samples, 0)
			users = torch.LongTensor(users).to(self.gpu)

		# Mask for Negative samples
		mask = batch_user_R.clone()
		mask[users, negative_samples] = 1.
		
		gt = batch_user_R[mask > 0.]
		pred = output[mask > 0.]

		loss = self.loss(pred, gt)

		return loss
