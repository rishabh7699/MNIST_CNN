import idx2numpy as inp
import tensorflow as tf

# loading dataset
file1  = "train-images.idx3-ubyte"
ndarr1 = inp.convert_from_file(file1)

file2  = "train-labels.idx1-ubyte"
ndarr2 = inp.convert_from_file(file2)

file3  = "t10k-images.idx3-ubyte"
ndarr3 = inp.convert_from_file(file3)

file4  = "t10k-labels.idx1-ubyte"
ndarr4 = inp.convert_from_file(file4)

c     = [1,16,32]
f     = [3,7]
lambd = 0.001
steps = 100
m     = 20000
t_m   = 1000
bsize = 100

# function to print accuracy
def print_accuracy(Input,Output,string,l):
  c=0
  res=s.run(tf.argmax(y4,1),feed_dict={inp: Input})
  for i in range(l):
    if res[i]==Output[i]:
       c+=1
  print(string+" accuracy is:",str((c/l)*100))


# modifying input accoring to batch size, traning set size and test set size
ndarr1 = ndarr1.reshape(60000,28,28,1)
ndarr3 = ndarr3.reshape(10000,28,28,1)

ndarr1 = ndarr1[0:m]
ndarr2 = ndarr2[0:m]
ndarr3 = ndarr3[0:t_m]
ndarr4 = ndarr4[0:t_m]

Ibatch=[]
Obatch=[]
for i in range(m//bsize):
    Ibatch.append(ndarr1[i*bsize:i*bsize+bsize])
    Obatch.append(ndarr2[i*bsize:i*bsize+bsize])

# initializing values for weight, filters and bias with random numbers
w=[]
b=[]
w.append(tf.Variable(tf.random_uniform(shape=(f[0],f[0],c[0],c[1])),dtype=tf.float32))
b.append(tf.Variable(tf.random_uniform(shape=(c[1],)),dtype=tf.float32))

w.append(tf.Variable(tf.random_uniform(shape=(f[1],f[1],c[1],c[2])),dtype=tf.float32))
b.append(tf.Variable(tf.random_uniform(shape=(c[2],)),dtype=tf.float32))

w.append(tf.Variable(tf.random_uniform(shape=(12800,1000)),dtype=tf.float32))
b.append(tf.Variable(tf.random_uniform(shape=(1000,)),dtype=tf.float32))

w.append(tf.Variable(tf.random_uniform(shape=(1000,10)),dtype=tf.float32))
b.append(tf.Variable(tf.random_uniform(shape=(10,)),dtype=tf.float32))

inp=tf.placeholder(tf.float32)
out=tf.placeholder(tf.int32)


# Model
y1_t = tf.nn.conv2d(input=inp ,filter=w[0],padding="VALID",strides=[1,1,1,1])
y1   = tf.nn.relu(y1_t + b[0])

y2_t = tf.nn.conv2d(input=y1,filter=w[1],padding="VALID",strides=[1,1,1,1])
y2   = tf.nn.relu(y2_t+b[1])
t    = tf.shape(y2)
y2   = tf.reshape(y2,shape=[t[0],t[1]*t[2]*t[3]])

y3   = tf.nn.relu(tf.matmul(y2,w[2])+b[2])
y4   = tf.matmul(y3,w[3])+b[3]


cost=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=out, logits=y4))
optimizer=tf.train.AdamOptimizer(lambd)
train=optimizer.minimize(cost)

init_op = tf.global_variables_initializer()
s=tf.Session()
s.run(init_op)

# running optimizer steps no. of times
for i in range(steps):
    print("\nstep no:",str(i))
    for j in range(m//bsize):
      s.run(train,feed_dict={inp:Ibatch[j] , out:Obatch[j]})  
    print_accuracy(ndarr1,ndarr2,"Train",m)
    print_accuracy(ndarr3,ndarr4,"Test",t_m)
    
s.close()
