import numpy as np
import pandas as pd
import library.train_utils

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, TimeDistributed
from keras.layers import Conv1D, Reshape, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras import backend as K
from sklearn import metrics

class ModelCommon(object):
	X_file = 'X.npy'
	Y_file = 'Y.npy'
	D_file = 'D.npy'
	tr_idx_file = 'tr_idx.npy'
	cv_idx_file = 'cv_idx.npy'

	@staticmethod
	def get_config_file(model_dir_path, mn):
		return model_dir_path + '/' + mn + '-config.npy'

	@staticmethod
	def get_weight_file(model_dir_path, mn):
		return model_dir_path + '/' + mn + '-weights.h5'

	@staticmethod
	def get_architecture_file(model_dir_path, mn):
		return model_dir_path + '/' + mn + '-architecture.json'

	@staticmethod
	def get_out_file(model_dir_path, mn):
		return model_dir_path + '/' + mn + '-out.csv'

class RocAucEvaluation(Callback):
	def __init__(self, validation_data=(), interval=1):
		super(Callback, self).__init__()

		self.interval = interval
		self.X_val, self.y_val = validation_data

	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.interval == 0:
			y_pred = self.model.predict(self.X_val, verbose=0)
			score = metrics.roc_auc_score(self.y_val, y_pred.reshape(-1))
			print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

class EventEmbLSTMConv1DNet(ModelCommon):
	model_name = 'event-emb-lstm-conv1d'
	VERBOSE = 1

	def __init__(self):
		self.model_name = EventEmbLSTMConv1DNet.model_name
		self.model = None
		self.metric = None
		self.threshold = 5.0
		self.config = None
		self.sequence_length = 200
		self.inp_shape=(self.sequence_length,) 
		self.alphabet_size = 252

	@staticmethod
	def create_model(metric_list, inp_shape, 
		alphabet_size,
		emb_size=64, 
		spatial_dropout = 0.2, 
		lstm_units = 100, 
		conv_filters = 64,
		conv_kernel = 3):
		inp = Input(shape=inp_shape)
		x = Embedding(alphabet_size, emb_size)(inp)
		x = SpatialDropout1D(spatial_dropout)(x)
		x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.1,recurrent_dropout=0.1))(x)
		x = Conv1D(conv_filters, kernel_size = conv_kernel, padding = "valid", kernel_initializer = "glorot_uniform")(x)
		avg_pool = GlobalAveragePooling1D()(x)
		max_pool = GlobalMaxPooling1D()(x)
		conc = concatenate([avg_pool, max_pool])
		dense_1 = Dense(32, activation="relu")(conc)
		outp = Dense(1, activation="sigmoid")(dense_1)
		
		model = Model(inputs=inp, outputs=outp)
		model.compile(loss='binary_crossentropy',
				  optimizer=Adam(lr=1e-3),
				  metrics=metric_list)
		#print(model.summary())
		return model

	def load_model(self, model_dir_path):
		self.model = self.create_model(metric_list=[self.metric], inp_shape = self.inp_shape, alphabet_size = self.alphabet_size)
		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		self.model.load_weights(weight_file_path)

	def fit(self, dataset_dir, model_dir_path, batch_size=128, epochs=3, metric='accuracy'):

		self.metric = metric

		X = np.load(dataset_dir + '/' + ModelCommon.X_file)
		Y = np.load(dataset_dir + '/' + ModelCommon.Y_file) / 2
		tr_idx = np.load(dataset_dir + '/' + ModelCommon.tr_idx_file)
		cv_idx = np.load(dataset_dir + '/' + ModelCommon.cv_idx_file)

		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		architecture_file_path = self.get_architecture_file(model_dir_path, self.model_name)
		
		checkpoint = ModelCheckpoint(weight_file_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
		roc = RocAucEvaluation(validation_data=(X[cv_idx], Y[cv_idx]), interval=1)
		self.model = self.create_model(metric_list=[self.metric], inp_shape = self.inp_shape, alphabet_size = self.alphabet_size)
		print(self.model.summary())
		open(architecture_file_path, 'w').write(self.model.to_json())

		print('\nModel ' + self.model_name + ' training is started')
		# training
		history = self.model.fit(x=X[tr_idx], y=Y[tr_idx],
								 batch_size=batch_size, epochs=epochs,
								 verbose=self.VERBOSE, validation_data=(X[cv_idx], Y[cv_idx]),
								 callbacks=[checkpoint, roc]).history

		print('Model ' + self.model_name + ' training is finished\n\n')
		

		print('Model ' + self.model_name + ' evaluation is started')
		self.load_model(model_dir_path)
		y_pred = self.model.predict(X[cv_idx], batch_size=1024, verbose=1)
		y_df = pd.DataFrame({'idx': cv_idx, 'y_true': Y[cv_idx], 'y_pred': y_pred.reshape(-1)})
		y_df.to_csv(self.get_out_file(model_dir_path, self.model_name))

		print('Model ' + self.model_name + ' evaluation is finished')

		print('estimated threshold is ' + str(self.threshold))

		roc_auc, precision, recall = library.train_utils.evaluate_prediction(Y[cv_idx], y_pred.reshape(-1), self.threshold)
		print("\nRESULT:\nROC AUC = {}, Precision = {}, Recall = {}\n".format(roc_auc, precision, recall))

		self.config = dict()
		self.config['metric'] = self.metric
		self.config['threshold'] = self.threshold
		self.config['inp_shape'] = self.inp_shape
		self.config['alphabet_size'] = self.alphabet_size
		self.config['roc_auc'] = roc_auc
		self.config['precision'] = precision
		self.config['recall'] = recall
		config_file_path = self.get_config_file(model_dir_path, self.model_name)
		np.save(config_file_path, self.config)

		return history

class IntervalLSTMConv1DNet(ModelCommon):
	model_name = 'interval-lstm-conv1d'
	VERBOSE = 1

	def __init__(self):
		self.model_name = IntervalLSTMConv1DNet.model_name
		self.model = None
		self.metric = None
		self.threshold = 5.0
		self.config = None
		self.sequence_length = 200
		self.inp_shape=(self.sequence_length,1) 

	@staticmethod
	def create_model(metric, inp_shape, 
		lstm_units = 100, 
		conv_filters = 64,
		conv_kernel = 3):
		inp = Input(shape=inp_shape)
		x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.1,recurrent_dropout=0.1))(inp)
		x = Conv1D(conv_filters, kernel_size = conv_kernel, padding = "valid", kernel_initializer = "glorot_uniform")(x)
		avg_pool = GlobalAveragePooling1D()(x)
		max_pool = GlobalMaxPooling1D()(x)
		conc = concatenate([avg_pool, max_pool])
		dense_1 = Dense(32, activation="relu")(conc)
		outp = Dense(1, activation="sigmoid")(dense_1)
		
		model = Model(inputs=inp, outputs=outp)
		model.compile(loss='binary_crossentropy',
				  optimizer=Adam(lr=1e-3),
				  metrics=[metric])
		#print(model.summary())
		return model

	def load_model(self, model_dir_path):
		self.model = self.create_model(metric=self.metric, inp_shape = self.inp_shape)
		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		self.model.load_weights(weight_file_path)

	def fit(self, dataset_dir, model_dir_path, batch_size=128, epochs=3, metric='accuracy'):

		self.metric = metric

		D = np.load(dataset_dir + '/' + ModelCommon.D_file)
		D = D.reshape((D.shape[0], D.shape[1], 1))
		Y = np.load(dataset_dir + '/' + ModelCommon.Y_file) / 2
		tr_idx = np.load(dataset_dir + '/' + ModelCommon.tr_idx_file)
		cv_idx = np.load(dataset_dir + '/' + ModelCommon.cv_idx_file)

		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		architecture_file_path = self.get_architecture_file(model_dir_path, self.model_name)
		
		checkpoint = ModelCheckpoint(weight_file_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
		roc = RocAucEvaluation(validation_data=(D[cv_idx], Y[cv_idx]), interval=1)
		self.model = self.create_model(metric=self.metric, inp_shape = self.inp_shape)
		print(self.model.summary())
		open(architecture_file_path, 'w').write(self.model.to_json())

		print('\nModel ' + self.model_name + ' training is started')
		# training
		history = self.model.fit(x=D[tr_idx], y=Y[tr_idx],
								 batch_size=batch_size, epochs=epochs,
								 verbose=self.VERBOSE, validation_data=(D[cv_idx], Y[cv_idx]),
								 callbacks=[checkpoint, roc]).history

		print('Model ' + self.model_name + ' training is finished\n\n')
		

		print('Model ' + self.model_name + ' evaluation is started')
		self.load_model(model_dir_path)
		y_pred = self.model.predict(D[cv_idx], batch_size=1024, verbose=1)
		y_df = pd.DataFrame({'idx': cv_idx, 'y_true': Y[cv_idx], 'y_pred': y_pred.reshape(-1)})
		y_df.to_csv(self.get_out_file(model_dir_path, self.model_name))

		print('Model ' + self.model_name + ' evaluation is finished')

		print('estimated threshold is ' + str(self.threshold))

		roc_auc, precision, recall = library.train_utils.evaluate_prediction(Y[cv_idx], y_pred.reshape(-1), self.threshold)
		print("\nRESULT:\nROC AUC = {}, Precision = {}, Recall = {}\n".format(roc_auc, precision, recall))

		self.config = dict()
		self.config['metric'] = self.metric
		self.config['threshold'] = self.threshold
		self.config['inp_shape'] = self.inp_shape
		self.config['roc_auc'] = roc_auc
		self.config['precision'] = precision
		self.config['recall'] = recall
		config_file_path = self.get_config_file(model_dir_path, self.model_name)
		np.save(config_file_path, self.config)

		return history

class EmbConcIntLSTMConv1DNet(ModelCommon):
	model_name = 'emb-conc-int-lstm-conv1d'
	VERBOSE = 1

	def __init__(self):
		self.model_name = EmbConcIntLSTMConv1DNet.model_name
		self.model = None
		self.metric = None
		self.threshold = 5.0
		self.config = None
		self.sequence_length = 200
		self.ev_inp_shape=(self.sequence_length,) 
		self.ts_inp_shape=(self.sequence_length,1) 
		self.alphabet_size = 252

	@staticmethod
	def create_model(metric, 
		ev_inp_shape,
		ts_inp_shape,
		alphabet_size,
		emb_size=64,
		lstm_units = 100, 
		conv_filters = 64,
		conv_kernel = 3):
		ev_inp = Input(shape=ev_inp_shape)
		ts_inp = Input(shape=ts_inp_shape)
		ev_emb = Embedding(alphabet_size, emb_size)(ev_inp)
		x = concatenate([ev_emb, ts_inp], axis=2)
		x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.1,recurrent_dropout=0.1))(x)
		x = Conv1D(conv_filters, kernel_size = conv_kernel, padding = "valid", kernel_initializer = "glorot_uniform")(x)
		avg_pool = GlobalAveragePooling1D()(x)
		max_pool = GlobalMaxPooling1D()(x)
		conc = concatenate([avg_pool, max_pool])
		dense_1 = Dense(32, activation="relu")(conc)
		outp = Dense(1, activation="sigmoid")(dense_1)
		
		model = Model(inputs=[ev_inp, ts_inp], outputs=outp)
		model.compile(loss='binary_crossentropy',
				  optimizer=Adam(lr=1e-3),
				  metrics=[metric])
		#print(model.summary())
		return model

	def load_model(self, model_dir_path):
		self.model = self.create_model(metric=self.metric, 
			ev_inp_shape = self.ev_inp_shape, 
			ts_inp_shape = self.ts_inp_shape, 
			alphabet_size = self.alphabet_size)
		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		self.model.load_weights(weight_file_path)

	def fit(self, dataset_dir, model_dir_path, batch_size=128, epochs=3, metric='accuracy'):

		self.metric = metric

		X = np.load(dataset_dir + '/' + ModelCommon.X_file)
		D = np.load(dataset_dir + '/' + ModelCommon.D_file)
		D = D.reshape((D.shape[0], D.shape[1], 1))
		Y = np.load(dataset_dir + '/' + ModelCommon.Y_file) / 2
		tr_idx = np.load(dataset_dir + '/' + ModelCommon.tr_idx_file)
		cv_idx = np.load(dataset_dir + '/' + ModelCommon.cv_idx_file)

		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		architecture_file_path = self.get_architecture_file(model_dir_path, self.model_name)
		
		checkpoint = ModelCheckpoint(weight_file_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
		roc = RocAucEvaluation(validation_data=([X[cv_idx], D[cv_idx]], Y[cv_idx]), interval=1)
		self.model = self.create_model(metric=self.metric, 
			ev_inp_shape = self.ev_inp_shape, 
			ts_inp_shape = self.ts_inp_shape, 
			alphabet_size = self.alphabet_size)
		print(self.model.summary())
		open(architecture_file_path, 'w').write(self.model.to_json())

		print('\nModel ' + self.model_name + ' training is started')
		# training
		history = self.model.fit(x=[X[tr_idx], D[tr_idx]], y=Y[tr_idx],
								 batch_size=batch_size, epochs=epochs,
								 verbose=self.VERBOSE, validation_data=([X[cv_idx], D[cv_idx]], Y[cv_idx]),
								 callbacks=[checkpoint, roc]).history

		print('Model ' + self.model_name + ' training is finished\n\n')
		

		print('Model ' + self.model_name + ' evaluation is started')
		self.load_model(model_dir_path)
		y_pred = self.model.predict([X[cv_idx], D[cv_idx]], batch_size=1024, verbose=1)
		y_df = pd.DataFrame({'idx': cv_idx, 'y_true': Y[cv_idx], 'y_pred': y_pred.reshape(-1)})
		y_df.to_csv(self.get_out_file(model_dir_path, self.model_name))

		print('Model ' + self.model_name + ' evaluation is finished')

		print('estimated threshold is ' + str(self.threshold))

		roc_auc, precision, recall = library.train_utils.evaluate_prediction(Y[cv_idx], y_pred.reshape(-1), self.threshold)
		print("\nRESULT:\nROC AUC = {}, Precision = {}, Recall = {}\n".format(roc_auc, precision, recall))

		self.config = dict()
		self.config['metric'] = self.metric
		self.config['threshold'] = self.threshold
		self.config['ev_inp_shape'] = self.ev_inp_shape
		self.config['ts_inp_shape'] = self.ts_inp_shape
		self.config['alphabet_size'] = self.alphabet_size
		self.config['roc_auc'] = roc_auc
		self.config['precision'] = precision
		self.config['recall'] = recall
		config_file_path = self.get_config_file(model_dir_path, self.model_name)
		np.save(config_file_path, self.config)

		return history

class EvtInt2RnnNet(ModelCommon):
	model_name = 'evt-int-2rnn-tdl'
	VERBOSE = 1

	def __init__(self):
		self.model_name = EvtInt2RnnNet.model_name
		self.model = None
		self.metric = None
		self.threshold = 5.0
		self.config = None
		self.sequence_length = 200
		self.ev_inp_shape=(self.sequence_length,) 
		self.ts_inp_shape=(self.sequence_length,1) 
		self.alphabet_size = 252

	@staticmethod
	def create_model(metric, 
		ev_inp_shape,
		ts_inp_shape,
		alphabet_size,
		emb_size=64,
		lstm_units = 100):
		ev_inp = Input(shape=ev_inp_shape)
		ts_inp = Input(shape=ts_inp_shape)


		x1 = Embedding(alphabet_size, emb_size)(ev_inp)
		x1 = SpatialDropout1D(0.2)(x1)
		x1 = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.1,recurrent_dropout=0.1))(x1)
		avg1 = GlobalAveragePooling1D()(x1)
		max1 = GlobalMaxPooling1D()(x1)
		avg1_re = Reshape((200, 1))(avg1)
		max1_re = Reshape((200, 1))(max1)
		
		
		x2 = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.1,recurrent_dropout=0.1))(ts_inp)
		avg2 = GlobalAveragePooling1D()(x2)
		max2 = GlobalMaxPooling1D()(x2)
		avg2_re = Reshape((200, 1))(avg2)
		max2_re = Reshape((200, 1))(max2)
		
		
		conc_t = concatenate([avg1_re, max1_re, avg2_re, max2_re], axis=2)
		tdl = TimeDistributed(Dense(16, activation='relu'))(conc_t)
		flatten = Flatten()(tdl)
		outp = Dense(1, activation="sigmoid")(flatten)
		
		model = Model(inputs=[ev_inp, ts_inp], outputs=outp)
		model.compile(loss='binary_crossentropy',
				  optimizer=Adam(lr=1e-3),
				  metrics=[metric])
		#print(model.summary())
		return model

	def load_model(self, model_dir_path):
		self.model = self.create_model(metric=self.metric, 
			ev_inp_shape = self.ev_inp_shape, 
			ts_inp_shape = self.ts_inp_shape, 
			alphabet_size = self.alphabet_size)
		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		self.model.load_weights(weight_file_path)

	def fit(self, dataset_dir, model_dir_path, batch_size=128, epochs=5, metric='accuracy'):

		self.metric = metric

		X = np.load(dataset_dir + '/' + ModelCommon.X_file)
		D = np.load(dataset_dir + '/' + ModelCommon.D_file)
		D = D.reshape((D.shape[0], D.shape[1], 1))
		Y = np.load(dataset_dir + '/' + ModelCommon.Y_file) / 2
		tr_idx = np.load(dataset_dir + '/' + ModelCommon.tr_idx_file)
		cv_idx = np.load(dataset_dir + '/' + ModelCommon.cv_idx_file)

		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		architecture_file_path = self.get_architecture_file(model_dir_path, self.model_name)
		
		checkpoint = ModelCheckpoint(weight_file_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
		roc = RocAucEvaluation(validation_data=([X[cv_idx], D[cv_idx]], Y[cv_idx]), interval=1)
		self.model = self.create_model(metric=self.metric, 
			ev_inp_shape = self.ev_inp_shape, 
			ts_inp_shape = self.ts_inp_shape, 
			alphabet_size = self.alphabet_size)
		print(self.model.summary())
		open(architecture_file_path, 'w').write(self.model.to_json())

		print('\nModel ' + self.model_name + ' training is started')
		# training
		history = self.model.fit(x=[X[tr_idx], D[tr_idx]], y=Y[tr_idx],
								 batch_size=batch_size, epochs=epochs,
								 verbose=self.VERBOSE, validation_data=([X[cv_idx], D[cv_idx]], Y[cv_idx]),
								 callbacks=[checkpoint, roc]).history

		print('Model ' + self.model_name + ' training is finished\n\n')
		

		print('Model ' + self.model_name + ' evaluation is started')
		self.load_model(model_dir_path)
		y_pred = self.model.predict([X[cv_idx], D[cv_idx]], batch_size=1024, verbose=1)
		y_df = pd.DataFrame({'idx': cv_idx, 'y_true': Y[cv_idx], 'y_pred': y_pred.reshape(-1)})
		y_df.to_csv(self.get_out_file(model_dir_path, self.model_name))

		print('Model ' + self.model_name + ' evaluation is finished')

		print('estimated threshold is ' + str(self.threshold))

		roc_auc, precision, recall = library.train_utils.evaluate_prediction(Y[cv_idx], y_pred.reshape(-1), self.threshold)
		print("\nRESULT:\nROC AUC = {}, Precision = {}, Recall = {}\n".format(roc_auc, precision, recall))

		self.config = dict()
		self.config['metric'] = self.metric
		self.config['threshold'] = self.threshold
		self.config['ev_inp_shape'] = self.ev_inp_shape
		self.config['ts_inp_shape'] = self.ts_inp_shape
		self.config['alphabet_size'] = self.alphabet_size
		self.config['roc_auc'] = roc_auc
		self.config['precision'] = precision
		self.config['recall'] = recall
		config_file_path = self.get_config_file(model_dir_path, self.model_name)
		np.save(config_file_path, self.config)

		return history

class EvtInt1RnnNet(ModelCommon):
	model_name = 'evt-int-1rnn'
	VERBOSE = 1

	def __init__(self):
		self.model_name = EvtInt1RnnNet.model_name
		self.model = None
		self.metric = None
		self.threshold = 5.0
		self.config = None
		self.sequence_length = 200
		self.ev_inp_shape=(self.sequence_length,) 
		self.ts_inp_shape=(self.sequence_length,1) 
		self.alphabet_size = 252

	@staticmethod
	def create_model(metric, 
		ev_inp_shape,
		ts_inp_shape,
		alphabet_size,
		emb_size=64,
		lstm_units = 100, 
		conv_filters = 64):
		ev_inp = Input(shape=ev_inp_shape)
		x1 = Embedding(alphabet_size, emb_size)(ev_inp)
		
		ts_inp = Input(shape=ts_inp_shape)
		x2 = Conv1D(conv_filters, kernel_size=1, padding = "valid", kernel_initializer = "glorot_uniform")(ts_inp)

		x = concatenate([x1, x2], axis=2)
		
		x = SpatialDropout1D(0.2)(x)
		x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.1,recurrent_dropout=0.1))(x)
		avgp = GlobalAveragePooling1D()(x)
		maxp = GlobalMaxPooling1D()(x)
		
		
		conc_t = concatenate([avgp, maxp], axis=1)
		outp = Dense(1, activation="sigmoid")(conc_t)
		
		model = Model(inputs=[ev_inp, ts_inp], outputs=outp)
		model.compile(loss='binary_crossentropy',
				  optimizer=Adam(lr=1e-3),
				  metrics=[metric])
		#print(model.summary())
		return model

	def load_model(self, model_dir_path):
		self.model = self.create_model(metric=self.metric, 
			ev_inp_shape = self.ev_inp_shape, 
			ts_inp_shape = self.ts_inp_shape, 
			alphabet_size = self.alphabet_size)
		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		self.model.load_weights(weight_file_path)

	def fit(self, dataset_dir, model_dir_path, batch_size=128, epochs=5, metric='accuracy'):

		self.metric = metric

		X = np.load(dataset_dir + '/' + ModelCommon.X_file)
		D = np.load(dataset_dir + '/' + ModelCommon.D_file)
		D = D.reshape((D.shape[0], D.shape[1], 1))
		Y = np.load(dataset_dir + '/' + ModelCommon.Y_file) / 2
		tr_idx = np.load(dataset_dir + '/' + ModelCommon.tr_idx_file)
		cv_idx = np.load(dataset_dir + '/' + ModelCommon.cv_idx_file)

		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		architecture_file_path = self.get_architecture_file(model_dir_path, self.model_name)
		
		checkpoint = ModelCheckpoint(weight_file_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
		roc = RocAucEvaluation(validation_data=([X[cv_idx], D[cv_idx]], Y[cv_idx]), interval=1)
		self.model = self.create_model(metric=self.metric, 
			ev_inp_shape = self.ev_inp_shape, 
			ts_inp_shape = self.ts_inp_shape, 
			alphabet_size = self.alphabet_size)
		print(self.model.summary())
		open(architecture_file_path, 'w').write(self.model.to_json())

		print('\nModel ' + self.model_name + ' training is started')
		# training
		history = self.model.fit(x=[X[tr_idx], D[tr_idx]], y=Y[tr_idx],
								 batch_size=batch_size, epochs=epochs,
								 verbose=self.VERBOSE, validation_data=([X[cv_idx], D[cv_idx]], Y[cv_idx]),
								 callbacks=[checkpoint, roc]).history

		print('Model ' + self.model_name + ' training is finished\n\n')
		

		print('Model ' + self.model_name + ' evaluation is started')
		self.load_model(model_dir_path)
		y_pred = self.model.predict([X[cv_idx], D[cv_idx]], batch_size=1024, verbose=1)
		y_df = pd.DataFrame({'idx': cv_idx, 'y_true': Y[cv_idx], 'y_pred': y_pred.reshape(-1)})
		y_df.to_csv(self.get_out_file(model_dir_path, self.model_name))

		print('Model ' + self.model_name + ' evaluation is finished')

		print('estimated threshold is ' + str(self.threshold))

		roc_auc, precision, recall = library.train_utils.evaluate_prediction(Y[cv_idx], y_pred.reshape(-1), self.threshold)
		print("\nRESULT:\nROC AUC = {}, Precision = {}, Recall = {}\n".format(roc_auc, precision, recall))

		self.config = dict()
		self.config['metric'] = self.metric
		self.config['threshold'] = self.threshold
		self.config['ev_inp_shape'] = self.ev_inp_shape
		self.config['ts_inp_shape'] = self.ts_inp_shape
		self.config['alphabet_size'] = self.alphabet_size
		self.config['roc_auc'] = roc_auc
		self.config['precision'] = precision
		self.config['recall'] = recall
		config_file_path = self.get_config_file(model_dir_path, self.model_name)
		np.save(config_file_path, self.config)

		return history

class LinearEmptyEvtsBaseline(ModelCommon):
	model_name = 'ln-empty-evt-baseline'
	VERBOSE = 1

	def __init__(self):
		self.model_name = EvtInt1RnnNet.model_name
		self.model = None
		self.metric = None
		self.threshold = 5.0
		self.config = None
		self.sequence_length = 300
		self.ev_inp_shape=(self.sequence_length,) 
		self.alphabet_size = 252

	@staticmethod
	def create_model(metric_list, inp_shape, 
		alphabet_size,
		emb_size=64, 
		spatial_dropout = 0.2, 
		lstm_units = 100, 
		conv_filters = 64,
		conv_kernel = 3):
		inp = Input(shape=inp_shape)
		x = Embedding(alphabet_size, emb_size)(inp)
		x = SpatialDropout1D(spatial_dropout)(x)
		x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.1,recurrent_dropout=0.1))(x)
		x = Conv1D(conv_filters, kernel_size = conv_kernel, padding = "valid", kernel_initializer = "glorot_uniform")(x)
		avg_pool = GlobalAveragePooling1D()(x)
		max_pool = GlobalMaxPooling1D()(x)
		conc = concatenate([avg_pool, max_pool])
		dense_1 = Dense(32, activation="relu")(conc)
		outp = Dense(1, activation="sigmoid")(dense_1)
		
		model = Model(inputs=inp, outputs=outp)
		model.compile(loss='binary_crossentropy',
				  optimizer=Adam(lr=1e-3),
				  metrics=metric_list)
		#print(model.summary())
		return model

	def load_model(self, model_dir_path):
		self.model = self.create_model(metric_list=[self.metric], inp_shape = self.inp_shape, alphabet_size = self.alphabet_size)
		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		self.model.load_weights(weight_file_path)

	def fit(self, dataset_dir, model_dir_path, batch_size=128, epochs=5, metric='accuracy'):

		self.metric = metric

		X = np.load(dataset_dir + '/' + ModelCommon.X_file)
		D = np.load(dataset_dir + '/' + ModelCommon.D_file)
		I = D[:, 1:] - D[:, :-1]
		I = np.concatenate((D[:, :1], I), axis=1)
		Y = np.load(dataset_dir + '/' + ModelCommon.Y_file) / 2
		tr_idx = np.load(dataset_dir + '/' + ModelCommon.tr_idx_file)
		cv_idx = np.load(dataset_dir + '/' + ModelCommon.cv_idx_file)

		print('\nBuilding a dilated sequence is started for model' + self.model_name)
		X_tmp = []
		for i in np.arange(X.shape[0]):
			if(i % 50000 == 0):
				print("{} records processed".format(i))
			dil_seq = self.buildDilatedSequence(X[i][-L[i]:], I[i][-L[i]:])
			X_tmp.append(dil_seq)
		X = sequence.pad_sequences(np.array(X_tmp), maxlen=self.sequence_length)
		print('\nBuilding a dilated sequence is finished for model' + self.model_name)

		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		architecture_file_path = self.get_architecture_file(model_dir_path, self.model_name)
		
		checkpoint = ModelCheckpoint(weight_file_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
		roc = RocAucEvaluation(validation_data=(X[cv_idx], Y[cv_idx]), interval=1)
		self.model = self.create_model(metric_list=[self.metric], inp_shape = self.inp_shape, alphabet_size = self.alphabet_size)
		print(self.model.summary())
		open(architecture_file_path, 'w').write(self.model.to_json())

		print('\nModel ' + self.model_name + ' training is started')
		# training
		history = self.model.fit(x=X[tr_idx], y=Y[tr_idx],
								 batch_size=batch_size, epochs=epochs,
								 verbose=self.VERBOSE, validation_data=(X[cv_idx], Y[cv_idx]),
								 callbacks=[checkpoint, roc]).history

		print('Model ' + self.model_name + ' training is finished\n\n')
		

		print('Model ' + self.model_name + ' evaluation is started')
		self.load_model(model_dir_path)
		y_pred = self.model.predict(X[cv_idx], batch_size=1024, verbose=1)
		y_df = pd.DataFrame({'idx': cv_idx, 'y_true': Y[cv_idx], 'y_pred': y_pred.reshape(-1)})
		y_df.to_csv(self.get_out_file(model_dir_path, self.model_name))

		print('Model ' + self.model_name + ' evaluation is finished')

		print('estimated threshold is ' + str(self.threshold))

		roc_auc, precision, recall = library.train_utils.evaluate_prediction(Y[cv_idx], y_pred.reshape(-1), self.threshold)
		print("\nRESULT:\nROC AUC = {}, Precision = {}, Recall = {}\n".format(roc_auc, precision, recall))

		self.config = dict()
		self.config['metric'] = self.metric
		self.config['threshold'] = self.threshold
		self.config['inp_shape'] = self.inp_shape
		self.config['alphabet_size'] = self.alphabet_size
		self.config['roc_auc'] = roc_auc
		self.config['precision'] = precision
		self.config['recall'] = recall
		config_file_path = self.get_config_file(model_dir_path, self.model_name)
		np.save(config_file_path, self.config)

		return history

	def zerosNumberLinearPerXHours(self, min_amt, period):
		return int(min_amt / (period * 60))

	def buildDilatedSequence(self, seq, intervals):
		res = []
		for i in np.arange(seq.shape[0]):
			empty_no = self.zerosNumberLinearPerXHours(intervals[i], 12)
			for j in np.arange(empty_no):
				res.append(0)
			res.append(seq[i])
		return res

class NonLinearEmptyEvtsBaseline(ModelCommon):
	model_name = 'nonln-empty-evt-baseline'
	VERBOSE = 1

	def __init__(self):
		self.model_name = EvtInt1RnnNet.model_name
		self.model = None
		self.metric = None
		self.threshold = 5.0
		self.config = None
		self.sequence_length = 300
		self.ev_inp_shape=(self.sequence_length,) 
		self.alphabet_size = 252

	@staticmethod
	def create_model(metric_list, inp_shape, 
		alphabet_size,
		emb_size=64, 
		spatial_dropout = 0.2, 
		lstm_units = 100, 
		conv_filters = 64,
		conv_kernel = 3):
		inp = Input(shape=inp_shape)
		x = Embedding(alphabet_size, emb_size)(inp)
		x = SpatialDropout1D(spatial_dropout)(x)
		x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.1,recurrent_dropout=0.1))(x)
		x = Conv1D(conv_filters, kernel_size = conv_kernel, padding = "valid", kernel_initializer = "glorot_uniform")(x)
		avg_pool = GlobalAveragePooling1D()(x)
		max_pool = GlobalMaxPooling1D()(x)
		conc = concatenate([avg_pool, max_pool])
		dense_1 = Dense(32, activation="relu")(conc)
		outp = Dense(1, activation="sigmoid")(dense_1)
		
		model = Model(inputs=inp, outputs=outp)
		model.compile(loss='binary_crossentropy',
				  optimizer=Adam(lr=1e-3),
				  metrics=metric_list)
		#print(model.summary())
		return model

	def load_model(self, model_dir_path):
		self.model = self.create_model(metric_list=[self.metric], inp_shape = self.inp_shape, alphabet_size = self.alphabet_size)
		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		self.model.load_weights(weight_file_path)

	def fit(self, dataset_dir, model_dir_path, batch_size=128, epochs=5, metric='accuracy'):

		self.metric = metric

		X = np.load(dataset_dir + '/' + ModelCommon.X_file)
		D = np.load(dataset_dir + '/' + ModelCommon.D_file)
		I = D[:, 1:] - D[:, :-1]
		I = np.concatenate((D[:, :1], I), axis=1)
		Y = np.load(dataset_dir + '/' + ModelCommon.Y_file) / 2
		tr_idx = np.load(dataset_dir + '/' + ModelCommon.tr_idx_file)
		cv_idx = np.load(dataset_dir + '/' + ModelCommon.cv_idx_file)

		print('\nBuilding a dilated sequence is started for model' + self.model_name)
		X_tmp = []
		for i in np.arange(X.shape[0]):
			if(i % 50000 == 0):
				print("{} records processed".format(i))
			dil_seq = self.buildDilatedSequence(X[i][-L[i]:], I[i][-L[i]:])
			X_tmp.append(dil_seq)
		X = sequence.pad_sequences(np.array(X_tmp), maxlen=self.sequence_length)
		print('\nBuilding a dilated sequence is finished for model' + self.model_name)

		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		architecture_file_path = self.get_architecture_file(model_dir_path, self.model_name)
		
		checkpoint = ModelCheckpoint(weight_file_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
		roc = RocAucEvaluation(validation_data=(X[cv_idx], Y[cv_idx]), interval=1)
		self.model = self.create_model(metric_list=[self.metric], inp_shape = self.inp_shape, alphabet_size = self.alphabet_size)
		print(self.model.summary())
		open(architecture_file_path, 'w').write(self.model.to_json())

		print('\nModel ' + self.model_name + ' training is started')
		# training
		history = self.model.fit(x=X[tr_idx], y=Y[tr_idx],
								 batch_size=batch_size, epochs=epochs,
								 verbose=self.VERBOSE, validation_data=(X[cv_idx], Y[cv_idx]),
								 callbacks=[checkpoint, roc]).history

		print('Model ' + self.model_name + ' training is finished\n\n')
		

		print('Model ' + self.model_name + ' evaluation is started')
		self.load_model(model_dir_path)
		y_pred = self.model.predict(X[cv_idx], batch_size=1024, verbose=1)
		y_df = pd.DataFrame({'idx': cv_idx, 'y_true': Y[cv_idx], 'y_pred': y_pred.reshape(-1)})
		y_df.to_csv(self.get_out_file(model_dir_path, self.model_name))

		print('Model ' + self.model_name + ' evaluation is finished')

		print('estimated threshold is ' + str(self.threshold))

		roc_auc, precision, recall = library.train_utils.evaluate_prediction(Y[cv_idx], y_pred.reshape(-1), self.threshold)
		print("\nRESULT:\nROC AUC = {}, Precision = {}, Recall = {}\n".format(roc_auc, precision, recall))

		self.config = dict()
		self.config['metric'] = self.metric
		self.config['threshold'] = self.threshold
		self.config['inp_shape'] = self.inp_shape
		self.config['alphabet_size'] = self.alphabet_size
		self.config['roc_auc'] = roc_auc
		self.config['precision'] = precision
		self.config['recall'] = recall
		config_file_path = self.get_config_file(model_dir_path, self.model_name)
		np.save(config_file_path, self.config)

		return history

	def zerosNumberNonLinear(self, min_amt):
	    if(min_amt < 60):
	        return 0
	    elif(min_amt < (60 * 6)):
	        return 1
	    elif(min_amt < (60 * 24)):
	        return 2
	    elif(min_amt < (60 * 24 *3)):
	        return 3
	    else:
	        return 4

	def buildDilatedSequence(self, seq, intervals):
		res = []
		for i in np.arange(seq.shape[0]):
			empty_no = self.zerosNumberNonLinear(intervals[i])
			for j in np.arange(empty_no):
				res.append(0)
			res.append(seq[i])
		return res

class IntensityEvt2RnnNet(ModelCommon):
	model_name = 'evt-intensity-2rnn'
	VERBOSE = 1

	def __init__(self):
		self.model_name = EvtInt1RnnNet.model_name
		self.model = None
		self.metric = None
		self.threshold = 5.0
		self.config = None
		self.sequence_length = 200
		self.intensity_sequence = 200
		self.ev_inp_shape=(self.sequence_length,) 
		self.int_inp_shape=(self.intensity_sequence,1) 
		self.alphabet_size = 252

	@staticmethod
	def create_model(metric, 
		ev_inp_shape,
		int_inp_shape,
		alphabet_size,
		emb_size=64,
		lstm_units = 100):
		ev_inp = Input(shape=ev_inp_shape)
		x1 = Embedding(alphabet_size, emb_size)(ev_inp)
		x1 = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.1,recurrent_dropout=0.1))(x1)
		avg1 = GlobalAveragePooling1D()(x1)
		max1 = GlobalMaxPooling1D()(x1)
		
		int_inp = Input(shape=int_inp_shape)
		x2 = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.1,recurrent_dropout=0.1))(int_inp)
		avg2 = GlobalAveragePooling1D()(x2)
		max2 = GlobalMaxPooling1D()(x2)

		conc_t = concatenate([avg1, max1, avg2, max2], axis=1)
		out1 = Dense(32, activation="relu")(conc_t)
		outp = Dense(1, activation="sigmoid")(out1)
		
		model = Model(inputs=[ev_inp, int_inp], outputs=outp)
		model.compile(loss='binary_crossentropy',
				  optimizer=Adam(lr=1e-3),
				  metrics=[metric])
		#print(model.summary())
		return model

	def load_model(self, model_dir_path):
		self.model = self.create_model(metric=self.metric, 
			ev_inp_shape = self.ev_inp_shape, 
			int_inp_shape = self.int_inp_shape, 
			alphabet_size = self.alphabet_size)
		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		self.model.load_weights(weight_file_path)

	def fit(self, dataset_dir, model_dir_path, batch_size=128, epochs=5, metric='accuracy'):

		self.metric = metric

		X = np.load(dataset_dir + '/' + ModelCommon.X_file)
		D = np.load(dataset_dir + '/' + ModelCommon.D_file)
		Y = np.load(dataset_dir + '/' + ModelCommon.Y_file) / 2
		tr_idx = np.load(dataset_dir + '/' + ModelCommon.tr_idx_file)
		cv_idx = np.load(dataset_dir + '/' + ModelCommon.cv_idx_file)

		I = []
		for i in np.arange(X.shape[0]):
			if(i % 50000 == 0):
				print("{} records processed".format(i))
			int_seq = self.buildIntensityFunction(D[i][-L[i]:], self.intensity_sequence, 60 * 12)
			I.append(int_seq)
		I = np.array(I)
		print("Intensity seq shape is {}".format(I.shape))


		weight_file_path = self.get_weight_file(model_dir_path, self.model_name)
		architecture_file_path = self.get_architecture_file(model_dir_path, self.model_name)
		
		checkpoint = ModelCheckpoint(weight_file_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
		roc = RocAucEvaluation(validation_data=([X[cv_idx], I[cv_idx]], Y[cv_idx]), interval=1)
		self.model = self.create_model(metric=self.metric, 
			ev_inp_shape = self.ev_inp_shape, 
			ts_inp_shape = self.int_inp_shape, 
			alphabet_size = self.alphabet_size)
		print(self.model.summary())
		open(architecture_file_path, 'w').write(self.model.to_json())

		print('\nModel ' + self.model_name + ' training is started')
		# training
		history = self.model.fit(x=[X[tr_idx], I[tr_idx]], y=Y[tr_idx],
								 batch_size=batch_size, epochs=epochs,
								 verbose=self.VERBOSE, validation_data=([X[cv_idx], D[cv_idx]], Y[cv_idx]),
								 callbacks=[checkpoint, roc]).history

		print('Model ' + self.model_name + ' training is finished\n\n')
		

		print('Model ' + self.model_name + ' evaluation is started')
		self.load_model(model_dir_path)
		y_pred = self.model.predict([X[cv_idx], I[cv_idx]], batch_size=1024, verbose=1)
		y_df = pd.DataFrame({'idx': cv_idx, 'y_true': Y[cv_idx], 'y_pred': y_pred.reshape(-1)})
		y_df.to_csv(self.get_out_file(model_dir_path, self.model_name))

		print('Model ' + self.model_name + ' evaluation is finished')

		print('estimated threshold is ' + str(self.threshold))

		roc_auc, precision, recall = library.train_utils.evaluate_prediction(Y[cv_idx], y_pred.reshape(-1), self.threshold)
		print("\nRESULT:\nROC AUC = {}, Precision = {}, Recall = {}\n".format(roc_auc, precision, recall))

		self.config = dict()
		self.config['metric'] = self.metric
		self.config['threshold'] = self.threshold
		self.config['ev_inp_shape'] = self.ev_inp_shape
		self.config['int_inp_shape'] = self.int_inp_shape
		self.config['alphabet_size'] = self.alphabet_size
		self.config['roc_auc'] = roc_auc
		self.config['precision'] = precision
		self.config['recall'] = recall
		config_file_path = self.get_config_file(model_dir_path, self.model_name)
		np.save(config_file_path, self.config)

		return history

	def buildIntensityFunction(intervals, bin_no, bin_size_min):
	    res = np.zeros(bin_no)
	    bin_map = intervals / bin_size_min
	    print(bin_map)
	    for bin_i in bin_map:
	        bi = int(min(bin_i, bin_no-1))
	        res[bi] = res[bi] + 1
	    return res
