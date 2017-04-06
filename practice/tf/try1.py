import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

sess = tf.Session()

# print(sess.run(tf.add(node1, node2)))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Can also use adder_node = a + b
adder_node = tf.add(a, b)

# print(sess.run(adder_node, {a:0, b:3}))
# print(sess.run(adder_node, {a:[3, 1], b:[5, 4]}))

add_and_triple = adder_node * 3
# print(sess.run(add_and_triple, {a:0, b:3}))
# print(sess.run(add_and_triple, {a:[3, 1], b:[5, 4]}))

W = tf.Variable([3.], tf.float32)
b = tf.Variable([-4.], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)
print("Value of W: ")
print(sess.run(W))
print(W.get_shape())
print(sess.run(b))
print(b.get_shape())
print(sess.run(linear_model, {x:[1,2,3,4]}))


print("Something interesting.....")
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
