import numpy as np 
from layers import *
from logistic import *
from svm import *
import optim
from solver import *
from softmax import *
from cnn import ConvNet
import pickle
import tensorflow as tf # This is to obtain MNist dataset

def logit_regression(data, data_test, Din):

	# simple layer (no hidden layer)
	print('Simple NN with logistic regression:')
	np.random.seed(598)
	model_logit_simple = LogisticClassifier(
		input_dim=Din,
		hidden_dim=None,
		weight_scale=0.5,
		reg=0.0,
		fwd_fun=relu_forward,
		bwd_fun=relu_backward
	)
	net_logit_simple = Solver(
		model=model_logit_simple, data=data,
		update_rule='sgd_momentum',
		optim_config={
			'learning_rate': 0.1
		},
		lr_decay=1.0,
		batch_size=20,
		num_epochs=400,
		verbose=False,
		print_every=10
	)
	print('	training...')
	net_logit_simple.train()
	print('	best val acc	: %f ' % net_logit_simple.best_val_acc)
	print('	test acc	: %f\n ' % net_logit_simple.check_accuracy(data_test['X'], data_test['y'],
																		  num_samples=1000))
	# 2 layers
	np.random.seed(598)
	print('2 layer NN with logistic regression:')
	model_logit = LogisticClassifier(
		input_dim=Din,
		hidden_dim=40,
		reg=0.0,
		fwd_fun=relu_forward,
		bwd_fun=relu_backward
	)
	net_logit = Solver(
		model=model_logit, data=data,
		update_rule='sgd_momentum',
		optim_config={
				'learning_rate': 0.01,
				},
		lr_decay=1.0,
		batch_size=30,
		num_epochs=200,
		verbose=False,
		print_every=10
	)
	print('	training...')
	net_logit.train()
	print('	best val acc	: %f ' % net_logit.best_val_acc)
	print('	test acc	: %f\n ' % net_logit.check_accuracy(data_test['X'], data_test['y'], num_samples=1000))

def svm_classification(data, data_test, Din):
	
	y_train = data['y_train']
	y_train[y_train==0] = -1
	data['y_train'] = y_train

	# 0.920
	# simple layer (no hidden layer)
	print('Simple NN with hinge loss:')
	np.random.seed(598)
	model_svm_simple = SVM(
		input_dim=Din,
		hidden_dim=None,
		weight_scale=0.1,
		reg=0.0,
		fwd_fun=relu_forward,
		bwd_fun=relu_backward
	)
	net_svm_simple = Solver(
		model=model_svm_simple, data=data,
		update_rule='sgd_momentum',
		optim_config={
			'learning_rate': 0.01
		},
		lr_decay=1.0,
		batch_size=10,
		num_epochs=400,
		verbose=False,
		print_every=10
	)
	print('	training...')
	net_svm_simple.train()
	print('	best val acc	: %f ' % net_svm_simple.best_val_acc)
	print('	test acc	: %f\n ' % net_svm_simple.check_accuracy(data_test['X'], 
											data_test['y'], num_samples=1000))
	
	# 0.924
	# 2 layers
	np.random.seed(598)
	print('2 layer NN with hinge loss:')
	model_svm = SVM(
		input_dim=Din,
		hidden_dim=20,
		weight_scale=0.1,
		reg=0.0,
		fwd_fun=relu_forward,
		bwd_fun=relu_backward
	)
	net_svm = Solver(
		model=model_svm, data=data,
		update_rule='sgd_momentum',
		optim_config={
				'learning_rate': 0.01,
				},
		lr_decay=1.0,
		batch_size=10,
		num_epochs=400,
		verbose=False,
		print_every=10
	)
	print('	training...')
	net_svm.train()
	print('	best val acc	: %f ' % net_svm.best_val_acc)
	print('	test acc	: %f\n ' % net_svm.check_accuracy(data_test['X'], 
									data_test['y'], num_samples=1000))

def softmax_classification(data, data_test):

	# 0.922
	# simple layer (no hidden layer)
	print('Simple NN with softmax loss:')
	np.random.seed(598)
	model_softmax_simple = SoftmaxClassifier(
		input_dim=data['X_train'].shape[1:],
		hidden_dim=None,
		num_classes=10,
		reg=0.0, 
		fwd_fun=relu_forward, 
		bwd_fun=relu_backward
	)
	net_softmax_simple = Solver(
		model=model_softmax_simple, data=data,
		update_rule='adam',
		optim_config={
			'learning_rate': 0.001
		},
		lr_decay=1.0,
		batch_size=100,
		num_epochs=50,
		verbose=True,
		print_every=100
	)
	print('	training...')
	net_softmax_simple.train()
	print('	best val acc	: %f ' % net_softmax_simple.best_val_acc)
	print('	test acc	: %f\n ' % net_softmax_simple.check_accuracy(data_test['X'], 
											data_test['y'], num_samples=1000))

	# 0.987
	# 2 layers (no hidden layer)
	print('2 layer NN with softmax loss:')
	np.random.seed(598)
	model_softmax_simple = SoftmaxClassifier(
		input_dim=data['X_train'].shape[1:],
		hidden_dim=600,
		num_classes=10,
		reg=0.0, 
		fwd_fun=relu_forward, 
		bwd_fun=relu_backward
	)
	net_softmax_simple = Solver(
		model=model_softmax_simple, data=data,
		update_rule='adam',
		optim_config={
			'learning_rate': 0.001
		},
		lr_decay=1.0,
		batch_size=100,
		num_epochs=20,
		verbose=True,
		print_every=100
	)
	print('	training...')
	net_softmax_simple.train()
	print('	best val acc	: %f ' % net_softmax_simple.best_val_acc)
	print('	test acc	: %f\n ' % net_softmax_simple.check_accuracy(data_test['X'], 
											data_test['y'], num_samples=1000))

def my_cnn(data, data_test):

	# 3 layers
	print('3 layer CNN with softmax loss:')
	np.random.seed(598)
	model_cnn = ConvNet(
		num_filters=32,
		filter_size=5,
		hidden_dim=512,
		num_classes=10,
		reg=0.0, 
		dropout = 0,
		normalization = True
	)
	net_cnn = Solver(
		model=model_cnn, data=data,
		update_rule='adam',
		optim_config={
			'learning_rate': 0.01
		},
		lr_decay=1.0,
		batch_size=50,
		num_epochs=15,
		verbose=True,
		print_every=1
	)
	print('	training...')
	net_cnn.train()
	print('	best val acc	: %f ' % net_cnn.best_val_acc)
	print('	test acc	: %f\n ' % net_cnn.check_accuracy(data_test['X'], 
											data_test['y'], num_samples=1000))

	# 3 layers
	print('3 layer CNN with softmax loss:')
	np.random.seed(598)
	model_cnn = ConvNet(
		num_filters=32,
		filter_size=5,
		hidden_dim=512,
		num_classes=10,
		reg=0.0, 
		dropout = 0.3,
		normalization = True
	)
	net_cnn = Solver(
		model=model_cnn, data=data,
		update_rule='adam',
		optim_config={
			'learning_rate': 0.01
		},
		lr_decay=1.0,
		batch_size=50,
		num_epochs=10,
		verbose=True,
		print_every=1
	)
	print('	training...')
	net_cnn.train()
	print('	best val acc	: %f ' % net_cnn.best_val_acc)
	print('	test acc	: %f\n ' % net_cnn.check_accuracy(data_test['X'], 
											data_test['y'], num_samples=1000))

def main():

	with open('data.pkl', 'rb') as file:
		data_ = pickle.load(file, encoding='latin1')
	X, y = data_
	D_train = 500; D_val = 250
	Din = X.shape[1]

	data = {
			'X_train': X[:D_train, :],
			'y_train': y[:D_train],
			'X_val': X[D_train:D_train+D_val, :],
			'y_val': y[D_train:D_train+D_val]
	}

	data_test = {
			'X': X[D_train+D_val:, :],
			'y': y[D_train+D_val:]
	}

	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	'''
	mask = np.random.choice(x_train.shape[0], 6000)
	mask2 = np.random.choice(x_test.shape[0], 1000)
	x_train = x_train[mask,:]
	y_train = y_train[mask]
	x_test = x_test[mask2,:]
	y_test = y_test[mask2]
	'''
	val_start = x_train.shape[0] // 10

	data_mnist = {
			'X_train': x_train[:-val_start],
			'y_train': y_train[:-val_start],
			'X_val': x_train[-val_start:],
			'y_val': y_train[-val_start:]
	}

	data_mnist_test = {
			'X': x_test,
			'y': y_test
	}

	##########################################################
	##     		   	Logistic Classifier 					##
	##					-label: (0,1)                       ##
	##########################################################

	logit_regression(data, data_test, Din)

	##########################################################
	##     		   	SVM Binary Classifier 					##
	##				   -label: (-1,1)                       ##
	##########################################################

	svm_classification(data, data_test, Din)

	##########################################################
	##     		   	Softmax multi-Classifier 				##
	##				   -label: (-1,1)                       ##
	##########################################################

	softmax_classification(data_mnist, data_mnist_test)

	##########################################################
	##     		   	CNN multi-Classifier 					##
	##				   -label: (-1,1)                       ##
	##########################################################

	my_cnn(data_mnist, data_mnist_test)


if __name__ == '__main__':
	main()