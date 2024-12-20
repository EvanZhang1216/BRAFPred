import tensorflow as tf
from model import  BertModel
from dataset_scaffold_random import Functional_Graph_Bert_Dataset
from dataset_scaffold_random import Graph_Regression_Dataset
import time
import os
import tensorflow.keras as keras
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

keras.backend.clear_session()
optimizer = tf.keras.optimizers.Adam(1e-4)


medium3 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 4, 'd_model': 256, 'path': 'medium3_weights','addH':True}

arch = medium3
num_layers = arch['num_layers']
num_heads =  arch['num_heads']
d_model =  arch['d_model']
addH = arch['addH']


dff = d_model*2
vocab_size =18
dropout_rate = 0.1

model = BertModel(num_layers=num_layers,d_model=d_model,dff=dff,num_heads=num_heads,vocab_size=vocab_size)
train_dataset, test_dataset = Functional_Graph_Bert_Dataset(path='raw_data_QSAR.csv',smiles_field='Smiles',addH=addH).get_data()
# train_dataset, test_dataset = Functional_Graph_Bert_Dataset(path='raw_data_QSAR.csv',smiles_field='Smiles',addH=addH).get_data()

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None,None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
]

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def train_step(x, adjoin_matrix,y, char_weight):
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    with tf.GradientTape() as tape:
        predictions = model(x,adjoin_matrix=adjoin_matrix,mask=mask,training=True)
        loss = loss_function(y,predictions,sample_weight=char_weight)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)
    train_accuracy.update_state(y,predictions,sample_weight=char_weight)


@tf.function(input_signature = train_step_signature)
def test_step(x, adjoin_matrix, y, char_weight):
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    predictions = model(x,adjoin_matrix=adjoin_matrix,mask=mask,training=False)
    test_accuracy.update_state(y,predictions,sample_weight=char_weight)


for epoch in range(20):
    start = time.time()
    train_loss.reset_states()

    for (batch, (x, adjoin_matrix ,y , char_weight)) in enumerate(train_dataset):
        train_step(x, adjoin_matrix, y , char_weight)

        if batch % 500 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, train_loss.result()))
            print('Accuracy: {:.4f}'.format(train_accuracy.result()))

    print(arch['path'] + '/bert_weights{}_{}.h5'.format(arch['name'], epoch+1))
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    print('Accuracy: {:.4f}'.format(train_accuracy.result()))
    model.save_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],epoch+1))
    print('Saving checkpoint')
