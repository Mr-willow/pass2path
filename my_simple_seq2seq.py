import tensorflow as tf
import numpy as np

chars_num = 93
epoch = 100
batch_size = 10
input_length = 5
out_put_length = 20
encoder_input_length = 10
vector_length = 32
dropout_prob = 0.5
learning_rate = 0.001


class Seq2seq(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.input_embed = tf.keras.layers.Embedding(input_dim=chars_num+3, output_dim=32)
        self.decoder_input_embed = tf.keras.layers.Embedding(input_dim=chars_num+3, output_dim=32)
        self.encoder_lstm = tf.keras.layers.LSTMCell(units=vector_length, dropout=dropout_prob)
        self.decoder_lstm = tf.keras.layers.LSTMCell(units=vector_length, dropout=dropout_prob)
        self.decoder_output_dense1 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.decoder_output_dense2 = tf.keras.layers.Dense(units=chars_num+3, activation=tf.nn.softmax)

    def call(self, inputs, targets=None, training=False):
        encoder_outputs = []
        outputs = []
        x = self.input_embed(inputs)  # batch_size, length, embedsize
        state = [tf.zeros(shape=(inputs.shape[0], vector_length)), tf.zeros(shape=(inputs.shape[0], vector_length))]
        for t in range(inputs.shape[1]):
            encoder_output, state = self.encoder_lstm(x[:, t, :], state, training=training)  # batch_size, vector_length    2 * batch_size * vector_length
            encoder_outputs.append(encoder_output)

        if training:
            targets = self.decoder_input_embed(targets)  # batch_size, length, embed_size
            for t in range(targets.shape[1]):
                decoder_output, state = self.decoder_lstm(targets[:, t, :], state)  # batch_size, vector_length
                output = self.decoder_output_dense1(decoder_output)  # batch_size, 64
                output = self.decoder_output_dense2(output)  # batch_size, char_num+3
                outputs.append(output)

        return tf.convert_to_tensor(outputs)

seq2seq = Seq2seq()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

x = [
    [1, 2, 3, 4, 5],
    [8, 8, 8, 8, 8],
    [6 ,5 ,4 ,3 ,2]
]  # 3 * 5
y = [
    [1, 2, 3],
    [8, 8, 8],
    [6 ,5 ,4]
]  # 3 * 3

x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.int32)

for epoch_index in range(epoch):
    with tf.GradientTape() as tape:
        y_true = tf.one_hot(y, depth=chars_num+3)
        y_pred = seq2seq(x, y, training=True)
        print(y_pred.shape, y_true.shape)
        loss = tf.keras.losses.categorical_crossentropy(y_pred=y_pred, y_true=y_true)
        loss = tf.reduce_mean(loss)
        print('epoch: %d, loss: %f' %(epoch_index, loss))
        grads = tape.gradient(loss, seq2seq.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, seq2seq.variables))

y = seq2seq(x, y, training=True)

print(y)
print(len(y), y[0].shape)

