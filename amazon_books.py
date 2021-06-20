import time

import scipy.sparse
from sklearn import metrics

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import os

sys.path.append('C:\\Users\\Adonis\\Documents\\uva msds\\graph mining\\project\\gcn')
# print(sys.path)

import layers.graph as lg
import utils.sparse as us

g = nx.read_gml('C:\\Users\\Adonis\\Documents\\uva msds\\graph mining\\project\\gcn\\R\\polbooks.gml')

node_attrs = nx.get_node_attributes(g,'value')
labels_let = node_attrs.values()
labels_r = []
for i in labels_let:
    if i == 'c':
        labels_r.append(1)
    elif i == 'l':
        labels_r.append(0)
    else:
        labels_r.append(2)

nx.draw(
    g,
    cmap=plt.get_cmap('viridis'),
    node_color=list(labels_r))

adj = nx.adj_matrix(g)

# Get important parameters of adjacency matrix
n_nodes = adj.shape[0]

# Some preprocessing
adj_tilde = adj + np.identity(n=adj.shape[0])
d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1/2)
d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
adj_norm_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(adj_norm))

# Features are just the identity matrix
feat_x = np.identity(n=adj.shape[0])
feat_x_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(feat_x))

# Semi-supervised
memberships = [m - 1
               for m in labels_r]

nb_classes = len(set(memberships))
targets = np.array([memberships], dtype=np.int32).reshape(-1)
one_hot_targets = np.eye(nb_classes)[targets]

# Pick one at random from each class
labels_to_keep = [np.random.choice(
    np.nonzero(one_hot_targets[:, c])[0]) for c in range(nb_classes)]

y_train = np.zeros(shape=one_hot_targets.shape,
                   dtype=np.float32)
y_val = one_hot_targets.copy()

train_mask = np.zeros(shape=(n_nodes,), dtype=np.bool)
val_mask = np.ones(shape=(n_nodes,), dtype=np.bool)

for l in labels_to_keep:
    y_train[l, :] = one_hot_targets[l, :]
    y_val[l, :] = np.zeros(shape=(nb_classes,))
    train_mask[l] = True
    val_mask[l] = False


# TensorFlow placeholders
ph = {
    'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_mat"),
    'x': tf.sparse_placeholder(tf.float32, name="x"),
    'labels': tf.placeholder(tf.float32, shape=(n_nodes, nb_classes)),
    'mask': tf.placeholder(tf.int32)}

l_sizes = [4, 4, 2, nb_classes]

o_fc1 = lg.GraphConvLayer(
    input_dim=feat_x.shape[-1],
    output_dim=l_sizes[0],
    name='fc1',
    activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=ph['x'], sparse=True)

o_fc2 = lg.GraphConvLayer(
    input_dim=l_sizes[0],
    output_dim=l_sizes[1],
    name='fc2',
    activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc1)

o_fc3 = lg.GraphConvLayer(
    input_dim=l_sizes[1],
    output_dim=l_sizes[2],
    name='fc3',
    activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc2)

o_fc4 = lg.GraphConvLayer(
    input_dim=l_sizes[2],
    output_dim=l_sizes[3],
    name='fc4',
    activation=tf.identity)(adj_norm=ph['adj_norm'], x=o_fc3)


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


with tf.name_scope('optimizer'):
    loss = masked_softmax_cross_entropy(
        preds=o_fc4, labels=ph['labels'], mask=ph['mask'])

    accuracy = masked_accuracy(
        preds=o_fc4, labels=ph['labels'], mask=ph['mask'])

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)

    opt_op = optimizer.minimize(loss)

feed_dict_train = {ph['adj_norm']: adj_norm_tuple,
                   ph['x']: feat_x_tuple,
                   ph['labels']: y_train,
                   ph['mask']: train_mask}

feed_dict_val = {ph['adj_norm']: adj_norm_tuple,
                 ph['x']: feat_x_tuple,
                 ph['labels']: y_val,
                 ph['mask']: val_mask}

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 300
save_every = 1

t = time.time()
outputs = {}

list_train_loss = []
list_train_acc = []
list_val_loss = []
list_val_acc = []

# Train model
for epoch in range(1,epochs+1):
    # Construct feed dictionary

    # Training step
    _, train_loss, train_acc = sess.run(
        (opt_op, loss, accuracy), feed_dict=feed_dict_train)
    
    list_train_loss.append(train_loss)
    list_train_acc.append(train_acc)

    if epoch % save_every == 0:
        # Validation
        val_loss, val_acc = sess.run((loss, accuracy), feed_dict=feed_dict_val)

        list_val_loss.append(val_loss)
        list_val_acc.append(val_acc)

        # Print results
        print("Epoch:", '%04d' % (epoch),
              "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_acc),
              "val_loss=", "{:.5f}".format(val_loss),
              "val_acc=", "{:.5f}".format(val_acc),
              "time=", "{:.5f}".format(time.time() - t))

        feed_dict_output = {ph['adj_norm']: adj_norm_tuple,
                            ph['x']: feat_x_tuple}

        output = sess.run(o_fc3, feed_dict=feed_dict_output)
        outputs[epoch] = output      
        
        
y_p = tf.argmax(o_fc4, 1)
y_pred = sess.run([y_p], feed_dict=feed_dict_output)
y_pred = y_pred[0]
y_pred_fixed = []
for i in y_pred:
    if i==0:
        y_pred_fixed.append(1)
    elif i==1:
        y_pred_fixed.append(2)
    else:
        y_pred_fixed.append(0)
    
print(y_pred_fixed)

y_true = np.asarray(labels_r)
print(y_true)

print(metrics.classification_report(y_true, y_pred_fixed))
            
node_positions = {o: {n: tuple(outputs[o][j])
                      for j, n in enumerate(nx.nodes(g))}
                  for o in outputs}
plot_titles = {o: 'epoch {o}'.format(o=o) for o in outputs}

# Two subplots, unpack the axes array immediately
f, axes = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True)

e = list(node_positions.keys())

# print(outputs[300])
# print(len(node_positions[300]))
print(o_fc4)


for i, ax in enumerate(axes.flat):
    pos = node_positions[e[(i)]*50]
    ax.set_title(plot_titles[e[(i)]*50])

    nx.draw(
        g,
        cmap=plt.get_cmap('viridis'),
        node_color=list(labels_r),
        pos=pos, ax=ax, node_size=100)

plt.show()

plt.plot(list(range(1,301)), list_train_loss, 'r', label='Training Loss')
plt.plot(list(range(1,301)), list_val_loss, label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


plt.plot(list(range(1,301)), list_train_acc, 'r', label='Training Accuracy')
plt.plot(list(range(1,301)), list_val_acc, label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

