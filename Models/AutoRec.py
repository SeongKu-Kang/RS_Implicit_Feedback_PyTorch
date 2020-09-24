import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoRec(nn.Module):
	def __init__(self, user_count, item_count, hidden_dim, gpu):
		"""
		Parameters
		----------
		user_count : int
		item_count : int
		hidden_dim : int
			bottleneck layer dimension
		gpu : if available
		"""
		super(AutoRec, self).__init__()
		self.user_count = user_count
		self.item_count = item_count

		self.user_list = torch.LongTensor([i for i in range(user_count)]).to(gpu)
		
		self.hidden_dim = hidden_dim
		self.Encoder = nn.Linear(self.item_count, self.hidden_dim)
		self.Decoder = nn.Linear(self.hidden_dim, self.item_count)

		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()

		self.gpu = gpu

		# user-item similarity type
		self.sim_type = 'AE'


	def forward(self, batch_user, batch_user_R):
		hidden_states = self.Encoder(batch_user_R)
		hidden_states = self.relu(hidden_states)
		output = self.Decoder(hidden_states)
		
		return output


	def get_loss(self, output, batch_user_R):

		tmp = (batch_user_R - output)
		mask = batch_user_R.clone()

		loss = ((tmp * mask) ** 2).sum(1).sum()
		return loss
