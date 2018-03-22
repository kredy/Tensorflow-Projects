'''

LeNet-5 like neural network using Tensorflow.

'''

# Importing the necessary libraries
from keras.datasets import mnist
import tensorflow as tf
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 


# Hyper parameters
batch_size = 128
learning_rate = 0.0001
epochs = 5
train_dev_split = 0.12


# Load the data
class LoadData():

	def __init__(self):
		pass

	# Load using Keras
	def loadFromDataset(self):
		
		(x,y), (x_test,y_test) = mnist.load_data()
		x = x/225.0
		x_test = x_test/225.0
		x = x.reshape(-1,28,28,1)
		x_test = x_test.reshape(-1,28,28,1)
		
		return x,y,x_test,y_test

	# Split into training and dev set
	def devSplit(self,x,y,train_dev_split):
		
		x_train,x_dev,y_train, y_dev = train_test_split(x,y,
			test_size=train_dev_split,random_state=42)
		
		return x_train,y_train,x_dev,y_dev

	# Create mini batches
	def batch(self,tot_size,x,y,batch_size):
		
		if batch_size<tot_size:
			num_batches = tot_size//batch_size
			last_batch_size = tot_size%batch_size
			batchx = []
			batchy = []
			if last_batch_size == 0:
				for i in range(num_batches):
					a = i*batch_size
					temp = x[a:a+batch_size,]
					batchx.append(temp)
				for i in range(num_batches):
					a = i*batch_size
					temp = y[a:a+batch_size,]
					#print(temp.shape)
					batchy.append(temp)
				return batchx, batchy
			else:
				for i in range(num_batches):
					a = i*batch_size
					temp = x[a:a+batch_size,]
					#print(temp.shape)
					batchx.append(temp)
				temp = x[-last_batch_size:]
				batchx.append(temp)
				for i in range(num_batches):
					a = i*batch_size
					temp = y[a:a+batch_size,]
					#print(temp.shape)
					batchy.append(temp)
				temp = y[-last_batch_size:]
				batchy.append(temp)
				return batchx,batchy
		
		else:
			print("Batch size is bigger than sample size!")
			return 1


# Input to the network graph
class InputGraph():

	def __init__(self,img_w,img_h,num_classes):
		
		self.img_w = img_w
		self.img_h = img_h
		self.num_classes = num_classes

	# Define placeholders
	def placeholders(self):
		
		with tf.name_scope('placeholder') as placeholders:
			X = tf.placeholder(tf.float32,shape=[None,self.img_w,self.img_h,1])
			Y = tf.placeholder(tf.int32,shape=[None,])
		
		return X,Y

	# OneHot encoding
	def oneHot(self,Y):
		
		with tf.name_scope('OneHotEncoding') as OneHotEncoding:
			Y_oh = tf.one_hot(Y,depth=self.num_classes)
		
		return Y_oh

	# Parameters
	def WeightsAndBiases(self):
		
		with tf.variable_scope('Weights') as weights:
			Weights = {
			'w1' : tf.get_variable('w1',shape=[5,5,1,6],
				initializer=tf.contrib.layers.xavier_initializer()),
			'w2' : tf.get_variable('w2',shape=[5,5,6,16],
				initializer=tf.contrib.layers.xavier_initializer()),
			'w3' : tf.get_variable('w3',shape=[400,120],
				initializer=tf.contrib.layers.xavier_initializer()),
			'w4' : tf.get_variable('w4',shape=[120,84],
				initializer=tf.contrib.layers.xavier_initializer()),
			'w_out' : tf.get_variable('w_out',shape=[84,10],
				initializer=tf.contrib.layers.xavier_initializer())
			}
		
		with tf.name_scope('Biases') as biases:
			Biases = {
			'b1' : tf.Variable(tf.zeros([6]),'b1'),
			'b2' : tf.Variable(tf.zeros([16]),'b2'),
			'b3' : tf.Variable(tf.zeros([120]),'b3'),
			'b4' : tf.Variable(tf.zeros([84]),'b4'),
			'b_out' : tf.Variable(tf.zeros([10]),'b_out')
			}
		
		return Weights,Biases


# Network architecture
class ConvNet():

	def __init__(self,X):
		self.X = X

	# Building the network
	def net(self,Weights,Biases):
		
		with tf.name_scope('Conv_1') as Conv_1:
			layer = tf.nn.conv2d(self.X,Weights['w1'],strides=[1,1,1,1],
				padding='SAME',name='Conv2D_1')
			layer = tf.nn.bias_add(layer,Biases['b1'])
			layer = tf.nn.relu(layer,name='Activation_1')
			layer = tf.nn.max_pool(layer,ksize=[1,2,2,1],strides=[1,2,2,1],
				padding='VALID',name='max_pool_1')
		
		with tf.name_scope('Conv_2') as Conv_2:
			layer = tf.nn.conv2d(layer,Weights['w2'],strides=[1,1,1,1],
				padding='VALID',name='Conv2D_2')
			layer = tf.nn.bias_add(layer,Biases['b2'])
			layer = tf.nn.relu(layer,name='Activation_2')
			layer = tf.nn.max_pool(layer,ksize=[1,2,2,1],strides=[1,2,2,1],
				padding='VALID',name='max_pool_2')
		
		with tf.name_scope('Flatten') as Flatten:
			layer = tf.layers.flatten(layer,name='Flatten_Layer')
		
		with tf.name_scope('Dense_1') as Dense_1:
			layer = tf.matmul(layer,Weights['w3'])
			layer = tf.nn.bias_add(layer,Biases['b3'])
			layer = tf.nn.relu(layer,name='Activation_3')
		
		with tf.name_scope('Dense_2') as Dense_2:
			layer = tf.matmul(layer,Weights['w4'])
			layer = tf.nn.bias_add(layer,Biases['b4'])
			layer = tf.nn.relu(layer,name='Activation_4')
		
		with tf.name_scope('Out_Layer') as Out_Layer:
			layer = tf.matmul(layer,Weights['w_out'])
			layer = tf.nn.bias_add(layer,Biases['b_out'])
		
		return layer

# Perform Training
class StartNN():

	def __init__(self,X,Y,batch_x,batch_y):
		self.X = X
		self.Y = Y
		self.batch_x = batch_x
		self.batch_y = batch_y

	# Cross entropy loss function
	def lossFunction(self,Y_hat,Y_oh):
		
		loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_oh,logits=Y_hat))
		loss_summary = tf.summary.scalar('Cross_Entropy_Loss', loss)
		
		return loss, loss_summary

	# Train, test and predict
	def trainAndTest(self,loss,loss_summary,acc,acc_summary,update_ops,
		learning_rate,epochs,Y_hat,Y_oh,x_dev,y_dev,x_test,y_test,x_val):
		
		train_step = tf.train.AdamOptimizer(
			learning_rate=learning_rate).minimize(loss)
		
		with tf.Session() as sess:
			sess.run([tf.global_variables_initializer(),
				tf.local_variables_initializer()])
			writer = tf.summary.FileWriter("./LeNet5_like(temp)", sess.graph)
			print('-'*50)
			
			for i in range(epochs):
				for x,y in zip(self.batch_x,self.batch_y):
					loss_summ, _ = sess.run([loss_summary,train_step],
						feed_dict={self.X:x,self.Y:y})
					writer.add_summary(loss_summ,i)
					# Training accuracy
					accr,acc_sum,up = sess.run([acc,acc_summary,update_ops],
						feed_dict={self.X:x,self.Y:y})
					writer.add_summary(acc_sum,i)
				
				# Dev set accuracy
				# The same could also be done using tensorboard
				dev_acc,up_dev = sess.run([acc,
					update_ops],feed_dict={self.X:x_dev,self.Y:y_dev})
				print('Dev Set accuracy (epoch {}): {}'.format(i+1,dev_acc))
				print('-'*50)
			
			# Test set accuracy
			# The same could also be done using tensorboard
			test_accr,up = sess.run([acc,update_ops],
			feed_dict={self.X:x_test,self.Y:y_test})
			print('Test set accuracy: {}'.format(test_accr))
			# Prediction
			prediciton  = sess.run(tf.argmax(Y_hat,1),feed_dict={self.X:x_val})
			print('Pred value: {}'.format(prediciton))


# Metrics
class Metrics():

	def __init__(self,Y_hat,Y_oh):
		self.Y_hat = Y_hat
		self.Y_oh = Y_oh

	# Accuracy
	def accuracyMetric(self):
		
		acc, update_ops = tf.metrics.accuracy(tf.argmax(self.Y_oh,1),
			tf.argmax(self.Y_hat,1))
		# corr_pred=tf.equal(tf.argmax(self.Y_oh,1),tf.argmax(self.Y_hat,1))
		# acc = tf.reduce_mean(tf.cast(corr_pred,tf.float32))
		acc_summary = tf.summary.scalar('Accuracy', acc)
		
		return acc, acc_summary, update_ops



# Main function
def main(learning_rate,batch_size,epochs,train_dev_split):

	loadData = LoadData()
	x,y,x_test,y_test = loadData.loadFromDataset()
	x_train,y_train,x_dev,y_dev = loadData.devSplit(x,y,train_dev_split)
	tot_size = x_train.shape[0]
	batch_x,batch_y = loadData.batch(tot_size,x_train,y_train,batch_size)
	# Change the value to predict a different number
	x_val = x_test[52]
	# Needed for matplotlib
	x_val1 = x_val.reshape(28,28)
	# Needed for tensorflow placeholder
	x_val = x_val.reshape(1,28,28,1)
	#print(x_val.shape)
	img_w = x_train.shape[1]
	img_h = x_train.shape[2]
	num_classes = 10

	inputs = InputGraph(img_w,img_h,num_classes)
	X,Y = inputs.placeholders()
	Y_oh = inputs.oneHot(Y)
	Weights,Biases = inputs.WeightsAndBiases()

	NNet = ConvNet(X)
	Y_hat = NNet.net(Weights,Biases)

	acc_metrics = Metrics(Y_hat,Y_oh)
	acc, acc_summary, update_ops = acc_metrics.accuracyMetric()

	training = StartNN(X,Y,batch_x,batch_y)
	loss, loss_summary = training.lossFunction(Y_hat,Y_oh)
	model = training.trainAndTest(loss,loss_summary,acc,acc_summary,update_ops,
		learning_rate,epochs,Y_hat,Y_oh,x_dev,y_dev,x_test,y_test,x_val)
	
	imgplot = plt.imshow(x_val1)
	plt.show()


if __name__ == '__main__':
	main(learning_rate,batch_size,epochs,train_dev_split)