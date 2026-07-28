import math
import torch


def evaluate(args, flag, dataset, model, item_pop):
	if flag == "valid":
		testDict = dataset.valid_dict
		stage_idx_all = torch.tensor([i for i in range(args.period+1)]).to(args.device)
		popularity = item_pop[:, :args.period+1]
	elif flag == "test":
		testDict = dataset.test_dict
		stage_idx_all = torch.tensor([i for i in range(args.period+2)]).to(args.device)
		popularity = item_pop[:, :args.period+2]

	model = model.eval()
	log_p_v_all = torch.log(popularity.sum(dim=1) / popularity.sum()).unsqueeze(-1)

	log_p_tv_list = []
	for start_i in minibatch(dataset.m_items, batch_size=args.batch_size):
		batch_items = list(range(start_i, min(start_i+args.batch_size, dataset.m_items)))
		batch_stage_gpu = (torch.ones_like(torch.tensor(batch_items))*8).to(args.device)
		batch_sub_pop = popularity[batch_items].unsqueeze(-1).int()
		batch_mask = torch.ones_like(batch_sub_pop).bool().to(args.device)
		with torch.no_grad():
			item_emb = model.item_embedding.weight[batch_items]
			log_p_tv = model.log_hawkes(item_emb, batch_stage_gpu, stage_idx_all, batch_sub_pop, batch_mask)
		log_p_tv_list.append(log_p_tv)
	log_p_tv_all = torch.concat(log_p_tv_list)

	true_list, pred_list = [], []
	for u in range(dataset.n_users):
		"""True Rating"""
		if len(testDict[u]) == 0:
			continue
		true_list.append(testDict[u])

		batch_users = torch.ones_like(log_p_tv_all).squeeze(-1).int()*u

		with torch.no_grad():
			user_emb = model.user_embedding(batch_users)
			item_emb = model.item_embedding.weight
			z_embed = torch.cat([user_emb, item_emb], axis=1)
			log_p_utv = model.y(z_embed) + log_p_tv_all * model.user_conformity

		pred = (log_p_utv + log_p_tv_all + log_p_v_all).squeeze(-1)


		"""Filtering item history indices"""
		exclude_items = list(dataset._allPos[u])
		if flag == "test":
			valid_items = dataset.getUserValidItems(torch.tensor([u])) # exclude validation items
			exclude_items.extend(valid_items)
		pred[exclude_items] = -(9999)
		_, pred_k = torch.topk(pred.squeeze(-1), k=max(args.topks))
		pred_list.append(pred_k.cpu())
	
	return true_list, pred_list


def minibatch(num, batch_size):
	for i in range(0, num, batch_size):
		yield i

        
def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        cnt = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
                cnt += 1

        precision.append(round(sumForPrecision / cnt, 4))
        recall.append(round(sumForRecall / cnt, 4))
        NDCG.append(round(sumForNdcg / cnt, 4))
        MRR.append(round(sumForMRR / cnt, 4))
        
    return precision, recall, NDCG, MRR
