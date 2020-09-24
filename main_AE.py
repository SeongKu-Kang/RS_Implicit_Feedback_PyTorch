import argparse

from Models.CDAE import CDAE

from Utils.dataset import implicit_CF_dataset_AE, implicit_CF_dataset_AE_test
from Utils.data_utils import read_LOO_settings

import torch
import torch.utils.data as data
import torch.optim as optim

from run import LOO_run_AE

def run():

	# gpu setting
	gpu = torch.device('cuda:' + str(opt.gpu))

	# for training
	model, lr, batch_size, num_ns = opt.model, opt.lr, opt.batch_size, opt.num_ns
	reg = opt.reg
	save = opt.save

	# dataset
	data_path, dataset, LOO_seed = opt.data_path, opt.dataset, opt.LOO_seed
	user_count, item_count, train_mat, train_interactions, valid_sample, test_sample, candidates = read_LOO_settings(data_path, dataset, LOO_seed)

	train_dataset = implicit_CF_dataset_AE(user_count, item_count, train_mat)
	test_dataset = implicit_CF_dataset_AE_test(user_count, item_count, train_mat, test_sample, valid_sample, candidates, batch_size)

	train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	# model
	if opt.model == 'CDAE':
		hidden_dim = opt.hidden_dim
		noise_level = opt.noise_level
		model = CDAE(user_count, item_count, hidden_dim, noise_level, num_ns, gpu)

	else:
		assert False

	# optimizer
	model = model.to(gpu)
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
	
	print("User::", user_count, "Item::", item_count, "Interactions::", len(train_interactions))

	# to save model
	model_save_path = 'None'
	if (save == 1):
		model_save_path = './Saved_models/' + opt.dataset +"/" + str(opt.model) +"_" + str(opt.lr) + "_" + str(opt.dim) + "_" + str(opt.reg) +'.model' + "_" + str(opt.LOO_seed)
	
	# start train
	LOO_run_AE(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# model
	parser.add_argument('--model', type=str, default='CDAE')
	parser.add_argument('--hidden_dim', type=int, default=60, help='bottleneck layer dimensions')
	parser.add_argument('--noise_level', type=float, default=0.2, help='for denoising AE')

	# training
	parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
	parser.add_argument('--reg', type=float, default=0.0001, help='for L2 regularization')
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--num_ns', type=int, default=5, help='number of negative samples')

	parser.add_argument('--gpu', type=int, default=0, help='0,1,2,3')

	parser.add_argument('--max_epoch', type=int, default=1000)
	parser.add_argument('--early_stop', type=int, default=20)
	parser.add_argument('--es_epoch', type=int, default=0, help='evaluation start epoch')
	parser.add_argument('--save', type=int, default=0, help='0: false, 1:true')

	# dataset
	parser.add_argument('--data_path', type=str, default='Dataset/')
	parser.add_argument('--dataset', type=str, default='citeULike', help='citeULike, Foursquare')
	parser.add_argument('--LOO_seed', type=int, default=0, help='0')

	opt = parser.parse_args()
	print(opt)

	run()