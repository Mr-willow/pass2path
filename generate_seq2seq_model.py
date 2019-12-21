import tensorflow as tf
import numpy as np
from data.generate_data import generate_dataset, get_path_vs_id, generate_bucket_list

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

path2id, id2path = get_path_vs_id()

chars_num = 52
path_num = len(path2id)
batch_size = 1
input_length = 5
out_put_length = 20
lstm_units = 128
embed_size = 32
vector_length = lstm_units
dropout_prob = 0.5
lstm_depth = 3
max_path_length = 5
beam_size = 100


class Seq2seq(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.epoch_recorder = tf.Variable(initial_value=0, name='epoch_index', dtype=tf.int32)
        self.input_embed = tf.keras.layers.Embedding(input_dim=chars_num+3, output_dim=embed_size)
        self.encoder_input_dense = tf.keras.layers.Dense(units=lstm_units, activation=tf.nn.relu)
        self.encoder_lstms = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(units=lstm_units, dropout=dropout_prob) for i in range(lstm_depth)])

        self.decoder_input_embed = tf.keras.layers.Embedding(input_dim=path_num, output_dim=path_num // 4)
        self.decoder_input_dense = tf.keras.layers.Dense(units=lstm_units, activation=tf.nn.relu)
        self.decoder_lstms = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(units=lstm_units, dropout=dropout_prob) for i in range(lstm_depth)])
        self.decoder_output_dense1 = tf.keras.layers.Dense(units=lstm_units*4, activation=tf.nn.relu)
        self.decoder_output_dense2 = tf.keras.layers.Dense(units=path_num, activation=tf.nn.relu)
        self.decoder_output_softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, inputs, targets=None, training=False, beam_search=False):
        encoder_outputs = []
        outputs = []
        x = self.input_embed(inputs)  # batch_size, length, embedsize
        state = [tf.zeros(shape=(inputs.shape[0], vector_length)), tf.zeros(shape=(inputs.shape[0], vector_length))]
        states = [state for i in range(lstm_depth)]
        for t in range(inputs.shape[1]):
            encoder_dense_output = self.encoder_input_dense(x[:, t, :])  # batch_size, lstm_units
            encoder_output, states = self.encoder_lstms(encoder_dense_output, states, training=training)  # batch_size, vector_length    2 * batch_size * vector_length
            encoder_outputs.append(encoder_output)

        if training:
            end_symbol = tf.multiply(tf.ones(shape=(targets.shape[0],), dtype=tf.int64), path2id[('i', ' ', 1)])
            output_symbol = tf.ones(shape=(targets.shape[0],), dtype=tf.int64)
            targets = self.decoder_input_embed(targets)  # batch_size, length, embed_size
            for t in range(targets.shape[1]):
                if all(tf.equal(output_symbol, end_symbol)):
                    break
                decoder_dense_output = self.decoder_input_dense(targets[:, t, :])  # batch_size, lstm_units
                decoder_output, states = self.decoder_lstms(decoder_dense_output, states)  # batch_size, vector_length
                output = self.decoder_output_dense1(decoder_output)  # batch_size, 64
                output = self.decoder_output_dense2(output)  # batch_size, path_num
                output_symbol = tf.convert_to_tensor(np.argmax(output, axis=1))
                # output = np.argmax(output, axis=1)  # one_hot decode
                outputs.append(output)

            outputs = tf.convert_to_tensor(outputs)
            outputs = tf.transpose(outputs, perm=(1, 0, 2))
            self.epoch_recorder = tf.add(self.epoch_recorder, tf.constant(1, dtype=tf.int32))  # epoch_index+1

            return outputs
        else:
            if beam_search:
                # init beam
                beam_dict = {}  # record the path, probability and state
                for i in range(batch_size):
                    beam_dict[i] = {}  # record the path, probability and state
                start_symbol = tf.ones(shape=(batch_size, 1))
                start_symbol = self.decoder_input_embed(start_symbol)  # batch_size, 1, path_num
                decoder_dense_output = self.decoder_input_dense(start_symbol[:, 0, :])  # batch_size, lstm_units
                decoder_output, states = self.decoder_lstms(decoder_dense_output, states)  # batch_size, vector_length
                output = self.decoder_output_dense1(decoder_output)  # batch_size, lstm_units*4
                output = self.decoder_output_dense2(output)  # batch_size, path_num
                output_num = output.numpy()  # batch_size, path_num
                output_sort = np.argsort(output_num, axis=-1)  # batch_size, path_num
                for batch_index in range(batch_size):
                    for i in range(path_num):
                        if output_sort[batch_index][i] >= (path_num - beam_size):  # in beam_size
                            beam_dict[batch_index][path_num - output_sort[batch_index][i] - 1] = {
                                'states': states, 'path': [1, i], 'value': i, 'prob': output_num[batch_index][i]}

                for l in range(max_path_length):
                    # count output
                    for batch_index in range(batch_size):
                        beam = [[]]
                        new_beam_dict = {}
                        beam_state = []
                        beam_length = path_num * beam_size
                        beam = np.array(beam)
                        for sort_index in range(beam_size):
                            state = beam_dict[batch_index][sort_index]['states']
                            x = tf.multiply(tf.ones(shape=(1, 1)), beam_dict[batch_index][sort_index]['value'])
                            x = self.decoder_input_embed(x)  # 1, 1, embed_size
                            decoder_dense_output = self.decoder_input_dense(x[:, 0, :])  # 1, lstm_units
                            decoder_output, state = self.decoder_lstms(decoder_dense_output, state)  # 1, vector_length
                            output = self.decoder_output_dense1(decoder_output)  # 1, lstm_units*4
                            output = self.decoder_output_dense2(output)  # 1, path_num
                            beam = np.hstack((beam, tf.multiply(output, beam_dict[batch_index][sort_index]['prob'])))
                            beam_state.append(state)
                        beam_sort = np.argsort(beam)
                        # update beam
                        for i in range(beam_length):
                            if beam_sort[0][i] >= (beam_length - beam_size):
                                value = i % path_num
                                num = i // path_num
                                new_beam_dict[beam_length - beam_sort[0][i] - 1] = {
                                    'states': beam_state[num],
                                    'path': beam_dict[batch_index][num]['path'].copy(),
                                    'value': value,
                                    'prob': beam[0][i]
                                }
                                new_beam_dict[beam_length - beam_sort[0][i] - 1]['path'].append(value)
                        beam_dict[batch_index] = new_beam_dict

                outputs = []
                for batch_index in range(batch_size):
                    outputs.append([])
                    for i in range(beam_size):
                        outputs[batch_index].append(beam_dict[batch_index][i]['path'])

                return outputs




            else:
                outputs = []
                output = tf.ones(shape=(inputs.shape[0], 1))
                end_symbol = tf.zeros(shape=(10,), dtype=tf.int64)
                output_symbol = tf.ones(shape=(10,), dtype=tf.int64)
                while not all(tf.equal(output_symbol, end_symbol)):
                    output = self.decoder_input_embed(output)  # batch_size, length, embed_size
                    decoder_output, states = self.decoder_lstms(output[:, 0, :], states)  # batch_size, vector_length
                    output = self.decoder_output_dense1(decoder_output)  # batch_size, 64
                    output = self.decoder_output_dense2(output)  # batch_size, char_num+3
                    output_symbol = tf.convert_to_tensor(np.argmax(output, axis=1))
                    outputs.append(output)

                outputs = tf.convert_to_tensor(outputs)
                outputs = tf.transpose(outputs, perm=(1, 0, 2))

                return outputs


if __name__ == '__main__':
    # seq2seq = Seq2seq()
    # checkpoint = tf.train.Checkpoint(seq2seq=seq2seq)
    # checkpoint.restore(tf.train.latest_checkpoint('seq2seq_model'))
    #
    # train_dataset_dict, target_dataset_dict = generate_bucket_list('data/csdn_dodonew_reuse_uniq.txt')
    # for epoch_index in range(1):
    #     # train use bucket
    #     for k in train_dataset_dict:
    #         print('==k:', k)
    #         train_dataset = train_dataset_dict[k]
    #         target_dataset = target_dataset_dict[k]
    #         dataset = tf.data.Dataset.from_tensor_slices((train_dataset, target_dataset))
    #         dataset = dataset.shuffle(buffer_size=batch_size * 100)
    #         dataset = dataset.batch(batch_size, drop_remainder=True)
    #         for x, y in dataset:
    #             with tf.GradientTape() as tape:
    #                 y_true = tf.one_hot(y, depth=path_num)
    #                 y_pred = seq2seq(x, y, training=False, beam_search=True)
    #                 print(np.argmax(y_pred, axis=-1))
    x = [[1, 2, 3, 4, 5]]
    x = np.array(x)
    x = tf.convert_to_tensor(x)
    seq2seq = Seq2seq()
    y = seq2seq(x, training=False, beam_search=True)
    print(y[0].__len__())

