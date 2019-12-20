import tensorflow as tf
import numpy as np
from generate_seq2seq_model import Seq2seq
from data.generate_data import generate_dataset, get_path_vs_id, generate_bucket_list

path2id, id2path = get_path_vs_id()

chars_num = 52
batch_size = 1
learning_rate = 0.001
path_num = len(path2id)

seq2seq = Seq2seq()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
checkpoint = tf.train.Checkpoint(seq2seq=seq2seq)
manager = tf.train.CheckpointManager(checkpoint, max_to_keep=5, directory='seq2seq_model', checkpoint_name='model.ckpt')

train_dataset_dict, target_dataset_dict = generate_bucket_list('data/csdn_dodonew_reuse_uniq.txt')

tensorboard = tf.summary.create_file_writer('seq2seq_tensorboard')
checkpoint.restore(tf.train.latest_checkpoint('seq2seq_model'))


def train(epoch):
    for epoch_index in range(epoch):
        # train use bucket
        for k in train_dataset_dict:
            print('==k:', k)
            train_dataset = train_dataset_dict[k]
            target_dataset = target_dataset_dict[k]
            dataset = tf.data.Dataset.from_tensor_slices((train_dataset, target_dataset))
            dataset = dataset.shuffle(buffer_size=batch_size * 100)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            for x, y in dataset:
                with tf.GradientTape() as tape:
                    y_true = tf.one_hot(y, depth=path_num)
                    y_pred = seq2seq(x, y, training=True)
                    current_length = y_pred.shape[1]
                    loss = tf.keras.losses.categorical_crossentropy(y_pred=y_pred, y_true=y_true[:, 0:current_length, :])
                    loss = tf.reduce_mean(loss)
                    epoch_index = seq2seq.epoch_recorder.numpy()  # get epoch_index from model
                    print('epoch: %d, loss: %f' % (epoch_index, loss))
                    with tensorboard.as_default():
                        tf.summary.scalar('loss', loss, step=epoch_index)
                    grads = tape.gradient(loss, seq2seq.variables)
                    optimizer.apply_gradients(grads_and_vars=zip(grads, seq2seq.variables))

    manager.save()


if __name__ == '__main__':
    epoch = 1
    train(epoch)
