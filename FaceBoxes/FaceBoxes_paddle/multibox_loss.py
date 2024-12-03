# coding:utf-8

import paddle

class MultiBoxLoss(paddle.nn.Layer):
	num_classes = 2
	def __init__(self):
		super(MultiBoxLoss,self).__init__()

	def log_sum_exp(self, x, y):
		xmax = x.max()
		log_sum_exp = paddle.log(paddle.sum(paddle.exp(x-xmax), 1, keepdim=True)) + xmax
		return log_sum_exp - paddle.where(y > 0, x[:,1].unsqueeze(1), x[:,0].unsqueeze(1))


	def hard_negative_mining(self,conf_loss, pos):
		'''
		conf_loss [N*num_anhcors,]
		pos [N,num_anhcors]
		return negative indice
		'''
		batch_size, num_boxes, _ = pos.shape
		conf_loss = paddle.where(pos.reshape((-1,1)).astype('bool'), paddle.zeros_like(conf_loss), conf_loss)#去掉正样本,the rest are neg conf_loss
		conf_loss = conf_loss.reshape((batch_size,-1))
		idx = paddle.argsort(conf_loss, axis=1, descending=True)
		rank = paddle.argsort(idx, axis=1) # the rank means origin number's rank in the position

		num_pos = paddle.sum(pos, axis=1)
		num_neg = paddle.clip(3*num_pos.astype('float32'), max=num_boxes-1)
		neg = rank < num_neg.expand_as(rank) # torch.ByteTensor
		return neg.astype('int64')

	def forward(self, loc_preds, loc_targets, conf_preds, conf_targets):
		'''
		loc_preds    [batch,num_anhcors,4]
		loc_targets  [batch,num_anhcors,4]
		conf_preds   [batch,num_anhcors,2]
		conf_targets [batch,num_anhcors]
		'''
		num_anhcors = conf_targets.shape[1]
		pos = (conf_targets > 0).astype('int64')  # the place > 0 is a face
		num_pos =  paddle.sum(pos, axis=1, keepdim=True) # num of faces in one image over the batch_size
		num_matched_boxes = num_pos.sum() # num of all faces

		# got the loc loss
		pos = paddle.reshape(pos,(pos.shape[0], num_anhcors, 1))
		pos_mask1 = paddle.nonzero(pos.expand_as(loc_preds))
		pos_loc_preds = paddle.gather_nd(loc_preds, pos_mask1).reshape((-1,4)) # filter the pos face
		pos_loc_targets = paddle.gather_nd(loc_targets, pos_mask1).reshape((-1,4)) # filter the pos face
		loc_criterion = paddle.nn.SmoothL1Loss()
		loc_loss = loc_criterion(pos_loc_preds, pos_loc_targets) * 4. # 4 is get the sum of four loc

		# got the conf loss
		conf_loss = self.log_sum_exp(paddle.reshape(conf_preds, (-1, self.num_classes)),
									paddle.reshape(conf_targets,(-1,1)))

		neg = self.hard_negative_mining(conf_loss, pos) # (16*num_anhcors, (16,num_anhcors))
		pos_mask = pos.expand_as(conf_preds)
		neg_mask = neg.unsqueeze(2).expand_as(conf_preds)
		mask = paddle.nonzero(pos_mask+neg_mask)

		pos_and_neg = paddle.nonzero(pos.squeeze(-1)+neg)
		preds = paddle.gather_nd(conf_preds, mask).reshape((-1, self.num_classes))
		targets = paddle.gather_nd(conf_targets, pos_and_neg).reshape((-1,1))
	
		conf_loss = paddle.nn.CrossEntropyLoss(reduction='sum')(preds, targets)
		conf_loss /= num_matched_boxes.astype('float32')
		return loc_loss, conf_loss, loc_loss+conf_loss
