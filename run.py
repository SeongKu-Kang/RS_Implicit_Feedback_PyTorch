import time
from copy import deepcopy

import torch
import torch.optim as optim

from Utils.evaluation import evaluation, LOO_print_result, print_final_result


def LOO_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path):

	max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch

	save = False
	if model_save_path != None:	
		save= True

	template = {'best_score':-999, 'best_result':-1, 'final_result':-1}
	eval_dict = {5: deepcopy(template), 10:deepcopy(template), 20:deepcopy(template), 'early_stop':0,  'early_stop_max':early_stop, 'final_epoch':0}

	# begin training
	for epoch in range(max_epoch):
		
		tic1 = time.time()
		train_loader.dataset.negative_sampling()
		epoch_loss = []
		
		for batch_user, batch_pos_item, batch_neg_item in train_loader:
			
			# Convert numpy arrays to torch tensors
			batch_user = batch_user.to(gpu)
			batch_pos_item = batch_pos_item.to(gpu)
			batch_neg_item = batch_neg_item.to(gpu)
			
			# Forward Pass
			model.train()
			output = model(batch_user, batch_pos_item, batch_neg_item)
			batch_loss = model.get_loss(output)
			epoch_loss.append(batch_loss)
			
			# Backward and optimize
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			
		epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
		toc1 = time.time()
		
		# evaluation
		if epoch < es_epoch:
			verbose = 25
		else:
			verbose = 1

		if epoch % verbose == 0:
			is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
			LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)
				
			if is_improved:
				if save:
					torch.save(model.state_dict(), model_save_path)

		if (eval_dict['early_stop'] >= eval_dict['early_stop_max']):
			break

	print("BEST EPOCH::", eval_dict['final_epoch'])
	print_final_result(eval_dict)



def LOO_run_AE(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path):

	max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch

	save = False
	if model_save_path != None:	
		save= True

	template = {'best_score':-999, 'best_result':-1, 'final_result':-1}
	eval_dict = {5: deepcopy(template), 10:deepcopy(template), 20:deepcopy(template), 'early_stop':0,  'early_stop_max':early_stop, 'final_epoch':0}

	# begin training
	for epoch in range(max_epoch):
		
		tic1 = time.time()
		epoch_loss = []
		
		for batch_user, batch_user_R in train_loader:
			
			# Convert numpy arrays to torch tensors
			batch_user = batch_user.to(gpu)
			batch_user_R = batch_user_R.to(gpu)
			
			# Forward Pass
			model.train()
			output = model(batch_user, batch_user_R)
			batch_loss = model.get_loss(output, batch_user_R)
			epoch_loss.append(batch_loss)
			
			# Backward and optimize
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			
		epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
		toc1 = time.time()
		
		# evaluation
		if epoch < es_epoch:
			verbose = 25
		else:
			verbose = 1

		if epoch % verbose == 0:
			is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
			LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)
				
			if is_improved:
				if save:
					torch.save(model.state_dict(), model_save_path)

		if (eval_dict['early_stop'] >= eval_dict['early_stop_max']):
			break

	print("BEST EPOCH::", eval_dict['final_epoch'])
	print_final_result(eval_dict)