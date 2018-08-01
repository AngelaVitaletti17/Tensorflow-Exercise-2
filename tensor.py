import tensorflow as tf

#This lesson starts with more fundamentals and will consist of various notes
 
 '''
 This practice comes from a website called Medium.comes
 '''
 
#Let's begin with some notes
'''
Something that was not explained very clearly to me was exactly what a 'model' 
is in machine learning or what training a model meant. This website has a very
clear definition to help better understand it, and make it more relatable:

When finding the line of best fit or linear regression for a dataset, the optimal
path can be difficult to find. To do this, you have to do what is called "gradient
descent," according to the site. This is basically going step by step finding the
most optimal path each time. So, you find the most optimal first step. Then move
to the second. Find the most optimal step there. This reminds me of various path-
finding algorithms, perhaps not the most advanced or preferred, but effective.

There are also varaibles in tensorflow that I didn't understand. These notes, also,
are kind of a rehash of the explanation on medium.com. It is mainly a way for me to
help myself learn, but if anyone finds these useful, so be it! Just know that my
source is medium.com and to check it out :)

Placeholder - Exactly as it sounds. When we're training, this will hold values.
			  It can be used, as well, so specifiy the amount of things we are
			  performing training on.
Variable - These are used to show the optimal values that optimize our regression.
		   They can have output, just like a placeholder, as well as the dependent
		   varaibles we have, like a placeholder
'''

#Let's do a small exercise
x = tf.placeholder(tf.float32, [None, 1]) 	#This will be house size
W = tf.Variable(tf.zeros([1,1])) 			#W is the coefficient in front of x. It is the slope
b = tf.Variable(tf.zeros([1]))				#Y intercept, at least in this example
y = tf.matmul(x,W)+basestring				#The actual function

y_ = tf.placeholder(tf.float32, [None,1]) 	#Represent a place to hold data that we will give to the model, in this case, house prices
cost = tf.reduce_sum(tf.pow((y_-y),2))		#Calculate the sum of the the difference of the actual data point and the best-fit point, squared

for i in range(100):						#We're going to create some data; we don't have any
	x_f = np.array([[i]])
	y_f = np.array([[2*i]])
