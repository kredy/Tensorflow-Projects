'''

LeNet-5 like neural network using Tensorflow.

'''

# Importing the necessary libraries
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split 


# Hyper parameters
batch_size = 128
learning_rate = 0.01
epochs = 1
train_dev_split = 0.12


# Load the data
class LoadData():


	def __init__(self,data_filename,labels_filename):
		
		# Load CIFAR-10 from a pickeld file
		# The same could also be done using keras.datasets
		self.data_filename = data_filename
		self.labels_filename = labels_filename
		with open(data_filename,'rb') as fo:
			x = pickle.load(fo)
		with open(labels_filename,'rb') as fo:
			y = pickle.load(fo)
		x = x/255.0
		y = y.reshape(-1)
		self.x = x
		self.y = y



	def train_dev_Set(self):

		# Spilliting into train and dev set
		x_train,x_dev,y_train,y_dev = train_test_split(self.x,self.y,
			test_size=0.15,random_state=25)
		self.y_train = y_train
		self.y_dev = y_dev
		
		return x_train, x_dev, y_train, y_dev


	def test_Set(self,test_data_filename,test_labels_filename):
		
		# Load test set from pickled file
		with open(test_data_filename, 'rb') as fo:
			x_test = pickle.load(fo)
		with open(test_labels_filename, 'rb') as fo:
			y_test = pickle.load(fo)
		x_test = x_test/255.0
		y_test = y_test.reshape(-1)
		self.y_test = y_test
		
		return x_test, y_test

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
			X = tf.placeholder(tf.float32,shape=[None,self.img_w,self.img_h,3])
			Y = tf.placeholder(tf.int32,shape=[None,])
			training = tf.placeholder(tf.float32)
		
		return X,Y,training

	# OneHot encoding
	def oneHot(self,Y):
		
		with tf.name_scope('OneHotEncoding') as OneHotEncoding:
			Y_oh = tf.one_hot(Y,depth=self.num_classes)
		
		return Y_oh

	# Parameters
	def WeightsAndBiases(self):
		
		Weights = {}

		def weights_add(name,shape):
			Weights[name] = tf.get_variable(name,shape=shape,
				initializer=tf.contrib.layers.xavier_initializer())

		with tf.variable_scope('Weights') as weights:
			Weights = {
			'w1' : tf.get_variable('w1',shape=[3,3,3,32],
				initializer=tf.contrib.layers.xavier_initializer()),
			}
			weights_add('w2',[2,2,32,32])
			weights_add('w3',[2,2,64,32])
			weights_add('w4',[2,2,32,32])
			weights_add('w5',[2,2,64,32])
			weights_add('w6',[2,2,32,32])
			weights_add('w7',[64,10])
			weights_add('w8',[10,10])
			weights_add('w_out',[10,10])
		
		with tf.name_scope('Biases') as biases:
			Biases = {
			'b7' : tf.Variable(tf.zeros([10]),'b7'),
			'b8' : tf.Variable(tf.zeros([10]),'b8'),
			'b_out' : tf.Variable(tf.zeros([10]),'b_out')
			}
		
		return Weights,Biases


# Blocks
class Blocks():

	def __init__(self,Weights,Biases):
		self.Weights = Weights
		self.Biases = Biases

	# Conv Block
	def ConvBlock(self,input_matrix,W_name,block_name):
		nxt_layer = tf.nn.conv2d(input_matrix,self.Weights[W_name],
			strides=[1,1,1,1],padding='SAME',name=block_name+'_Conv2D')
		# Easiler Batch Norm using a higher level API
		nxt_layer = tf.layers.batch_normalization(nxt_layer,name=block_name+'_BatchNorm')
		nxt_layer = tf.nn.leaky_relu(nxt_layer,alpha=0.001,name=block_name+'Activation')
		return nxt_layer

	# Dense Block
	def DenseBlock(self,input_matrix,W_name,B_name,block_name,keep_prob,out=False):
		nxt_layer = tf.matmul(input_matrix,self.Weights[W_name])
		nxt_layer = tf.nn.bias_add(nxt_layer,self.Biases[B_name])
		if out:
			return nxt_layer
		else:
			nxt_layer = tf.layers.batch_normalization(nxt_layer,name=block_name+'_BatchNorm')
			nxt_layer = tf.nn.leaky_relu(nxt_layer,alpha=0.001,name=block_name+'Activation')
			nxt_layer = tf.nn.dropout(nxt_layer,keep_prob=keep_prob,name=block_name+'Dropout')
			return nxt_layer


# Network architecture
class DenseNet():

	def __init__(self,X,blocks,training):
		self.X = X
		self.blocks = blocks
		self.training = training

	# Building the network
	def net(self,Weights,Biases):
		
		with tf.name_scope('Conv_1') as graph:
			con_layer = tf.nn.conv2d(self.X,Weights['w1'],strides=[1,1,1,1],
				padding='SAME',name='Conv2D_1')
			con_layer = tf.layers.batch_normalization(con_layer,name='BatchNorm_1')
			con_layer = tf.nn.leaky_relu(con_layer,alpha=0.001,name='Activation_1')
		
		with tf.name_scope('Block_1') as graph:
			layer = self.blocks.ConvBlock(con_layer,'w2','Block_1')
			layer = tf.concat([layer,con_layer],axis=3)

		with tf.name_scope('MaxPool_1') as graph:
			layer = tf.nn.max_pool(layer,[1,2,2,1],[1,2,2,1],'VALID',name='MaxPool_A')

		with tf.name_scope('Block_2') as graph:
			con_layer = self.blocks.ConvBlock(layer,'w3','Block_2A')
			layer = self.blocks.ConvBlock(con_layer,'w4','Block_2B')
			layer = tf.concat([layer,con_layer],axis=3)

		with tf.name_scope('MaxPool_2') as graph:
			layer = tf.nn.max_pool(layer,[1,2,2,1],[1,2,2,1],'VALID',name='MaxPool_B')

		with tf.name_scope('Block_3') as graph:
			con_layer = self.blocks.ConvBlock(layer,'w5','Block_3A')
			layer = self.blocks.ConvBlock(con_layer,'w6','Block_3B')
			layer = tf.concat([layer,con_layer],axis=3)

		with tf.name_scope('MaxPool_3') as graph:
			layer = tf.nn.max_pool(layer,[1,2,2,1],[1,2,2,1],'VALID',name='MaxPool_C')

		# Keras GlobalAvgPool is very simple to implement
		with tf.name_scope('Global_Pool') as graph:
			layer = tf.keras.layers.GlobalAveragePooling2D()(layer)
		
		with tf.name_scope('FC_1') as graph:
			layer = self.blocks.DenseBlock(layer,'w7','b7','Dense_1',self.training)

		with tf.name_scope('FC_2') as graph:
			layer = self.blocks.DenseBlock(layer,'w8','b8','Dense_2',self.training)

		with tf.name_scope('Out') as graph:
			layer = self.blocks.DenseBlock(layer,'w_out','b_out','Out_Layer',
				self.training,out=True)
		
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
		learning_rate,epochs,Y_hat,Y_oh,x_dev,y_dev,x_test,y_test,Training):
		
		train_step = tf.train.AdamOptimizer(
			learning_rate=learning_rate).minimize(loss)
		
		with tf.Session() as sess:
			sess.run([tf.global_variables_initializer(),
				tf.local_variables_initializer()])
			writer = tf.summary.FileWriter("./DenseNet_like(temp)", sess.graph)
			print('-'*50)
			
			for i in range(epochs):
				for x,y in zip(self.batch_x,self.batch_y):
					loss_summ, _ = sess.run([loss_summary,train_step],
						feed_dict={self.X:x,self.Y:y,Training:0.5})
					writer.add_summary(loss_summ,i)
					# Training accuracy
					accr,acc_sum,up = sess.run([acc,acc_summary,update_ops],
						feed_dict={self.X:x,self.Y:y,Training:1})
					writer.add_summary(acc_sum,i)
					print(accr)
				
				# Dev set accuracy
				# The same could also be done using tensorboard
				print('Finished Training')
				lst = []
				j = 0
				for x,y in zip(x_dev,y_dev):
					j = j+1
					dev_acc,up_dev = sess.run([acc,
						update_ops],feed_dict={self.X:x,self.Y:y,Training:1})
					print('Dev Set accuracy (Batch {}): {}'.format(j,dev_acc))
					print('-'*50)
					lst.append(dev_acc)
				print('Dev Set accuracy (epoch {}): {}'.format(i+1,sum(lst)/len(lst)))
			
			# Test set accuracy
			# The same could also be done using tensorboard
			lst = []
			for x,y in zip(x_test,y_test):
				test_accr,up = sess.run([acc,update_ops],
				feed_dict={self.X:x,self.Y:y,Training:1})
				print('Test set (Batch) accuracy: {}'.format(test_accr))
				lst.append(dev_acc)
			print('Test Set accuracy: {}'.format(sum(lst)/len(lst)))


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

	data = LoadData('./data.bin','./labels.bin')
	x_train, x_dev, y_train, y_dev = data.train_dev_Set()
	x_test, y_test = data.test_Set('./data_test.bin','./labels_test.bin')
	tot_size = x_train.shape[0]
	batch_x,batch_y = data.batch(tot_size,x_train,y_train,batch_size)
	tot_size_dev = x_dev.shape[0]
	batch_x_dev,batch_y_dev = data.batch(tot_size_dev,x_dev,y_dev,batch_size)
	tot_size_test = x_test.shape[0]
	batch_x_test,batch_y_test = data.batch(tot_size_test,x_test,y_test,batch_size)
	
	img_w = x_train.shape[1]
	img_h = x_train.shape[2]
	num_classes = 10

	inputs = InputGraph(img_w,img_h,num_classes)
	X,Y,Training = inputs.placeholders()
	Y_oh = inputs.oneHot(Y)
	Weights,Biases = inputs.WeightsAndBiases()

	blocks = Blocks(Weights,Biases)

	NNet = DenseNet(X,blocks,Training)
	Y_hat = NNet.net(Weights,Biases)

	acc_metrics = Metrics(Y_hat,Y_oh)
	acc, acc_summary, update_ops = acc_metrics.accuracyMetric()

	training = StartNN(X,Y,batch_x,batch_y)
	loss, loss_summary = training.lossFunction(Y_hat,Y_oh)
	model = training.trainAndTest(loss,loss_summary,acc,acc_summary,update_ops,
		learning_rate,epochs,Y_hat,Y_oh,
		batch_x_dev,batch_y_dev,batch_x_test,batch_y_test,Training)


if __name__ == '__main__':
	main(learning_rate,batch_size,epochs,train_dev_split)