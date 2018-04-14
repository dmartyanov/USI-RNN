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
from keras import backend as K

class ModelsCommon(object):
	X_file = 'X.npy'
	Y_file = 'Y.npy'
	D_file = 'D.npy'
	tr_idx_file = 'tr_idx.npy'
	cv_idx_file = 'cv_idx.npy'

class RocAucEvaluation(Callback):
	def __init__(self, validation_data=(), interval=1):
		super(Callback, self).__init__()

		self.interval = interval
		self.X_val, self.y_val = validation_data

	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.interval == 0:
			y_pred = self.model.predict(self.X_val, verbose=0)
			score = roc_auc_score(self.y_val, y_pred)
			print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

class EventEmbLSTMConv1DNet(object):
	model_name = 'event-emb-lstm-conv1d'
	VERBOSE = 1


	def __init__(self):
		self.model = None
		self.metric = None
		self.threshold = 5.0
		self.config = None
		self.sequence_length = 200
		self.inp_shape=(self.sequence_length,) 
		self.alphabet_size = 252

	@staticmethod
	def create_model(metric, inp_shape, 
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
				  metrics=[metric])
		print(model.summary())
		return model

	@staticmethod
	def get_config_file(model_dir_path):
		return model_dir_path + '/' + EventEmbLSTMConv1DNet.model_name + '-config.npy'

	@staticmethod
	def get_weight_file(model_dir_path):
		return model_dir_path + '/' + EventEmbLSTMConv1DNet.model_name + '-weights.h5'

	@staticmethod
	def get_architecture_file(model_dir_path):
		return model_dir_path + '/' + EventEmbLSTMConv1DNet.model_name + '-architecture.json'

	@staticmethod
	def get_out_file(model_dir_path):
		return model_dir_path + '/' + EventEmbLSTMConv1DNet.model_name + '-out.csv'

	def load_model(self, model_dir_path):
		config_file_path = self.get_config_file(model_dir_path)
		self.config = np.load(config_file_path).item()
		self.metric = self.config['metric']
		self.time_window_size = self.config['time_window_size']
		self.threshold = self.config['threshold']
		self.model = self.create_model(self.time_window_size, self.metric)
		weight_file_path = self.get_weight_file(model_dir_path)
		self.model.load_weights(weight_file_path)

	def fit(self, dataset_dir, model_dir_path, batch_size=128, epochs=4, metric='accuracy'):

		self.metric = metric

		X = np.load(dataset_dir + '/' + ModelsCommon.X_file)
		Y = np.load(dataset_dir + '/' + ModelsCommon.Y_file)
		tr_idx = np.load(dataset_dir + '/' + ModelsCommon.tr_idx_file)
		cv_idx = np.load(dataset_dir + '/' + ModelsCommon.cv_idx_file)

		weight_file_path = self.get_weight_file(model_dir_path=model_dir_path)
		architecture_file_path = self.get_architecture_file(model_dir_path)
		
		checkpoint = ModelCheckpoint(weight_file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
		roc = RocAucEvaluation(validation_data=(X[cv_idx], Y[cv_idx]), interval=1)
		self.model = self.create_model(metric=self.metric, inp_shape = self.inp_shape, alphabet_size = self.alphabet_size)
		open(architecture_file_path, 'w').write(self.model.to_json())

		print('Model ' + EventEmbLSTMConv1DNet.model_name + ' training is started')
		# training
		history = self.model.fit(x=X[tr_idx], y=Y[tr_idx],
								 batch_size=batch_size, epochs=epochs,
								 verbose=self.VERBOSE, validation_data=(X[cv_idx], Y[cv_idx]),
								 callbacks=[checkpoint, roc]).history

		print('Model ' + EventEmbLSTMConv1DNet.model_name + ' training is finished')
		

		print('Model ' + EventEmbLSTMConv1DNet.model_name + ' evaluation is started')
		self.model = self.load_model(model_dir_path)
		y_pred = model.predict(X[cv_idx], batch_size=1024, verbose=1)
		y_df = pd.DataFrame({'idx': cv_idx, 'y_true': Y[cv_idx], 'y_pred': y_pred.reshape(-1)})
		y_df.to_csv(self.get_out_file(model_dir_path))

		print('Model ' + EventEmbLSTMConv1DNet.model_name + ' evaluation is finished')

		print('estimated threshold is ' + str(self.threshold))

		roc_auc, precision, recall = train_utils.evaluate_prediction(model, Y[cv_idx], y_pred, self.threshold)

		self.config = dict()
		self.config['metric'] = self.metric
		self.config['threshold'] = self.threshold
		self.config['roc_auc'] = roc_auc
		self.config['precision'] = precision
		self.config['recall'] = recall
		config_file_path = self.get_config_file(model_dir_path=model_dir_path)
		np.save(config_file_path, self.config)

		return history