from losses.l1_loss import l1_loss_func
from losses.l2_loss import l2_loss_func
#from losses.l1_perceptual_loss import l1_perceptual_loss


def loss_func(output, target, loss):
	assert loss in ['l1', 'l2']
	if loss == 'l1':
		return l1_loss_func(output, target)
	elif loss == 'l2':
		return l2_loss_func(output, target)

