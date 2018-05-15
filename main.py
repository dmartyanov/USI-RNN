import os
import shutil

from library.models import EventEmbLSTMConv1DNet
from library.models import IntervalLSTMConv1DNet
from library.models import EmbConcIntLSTMConv1DNet
from library.models import EvtInt2RnnNet
from library.models import EvtInt1RnnNet
from library.models import LinearEmptyEvtsBaseline
from library.models import NonLinearEmptyEvtsBaseline
from library.models import IntensityEvt2RnnNet

DO_TRAINING = True
P = 200
A = 252

def recreateDirectory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        try:
            shutil.rmtree(dir)
            os.makedirs(dir)
        except OSError as e:
            print ("Error: {} - {}.".format(e.filename,e.strerror))

def startModel(model, model_dir, data_dir):
    recreateDirectory(model_dir)
    model.fit(data_dir, model_dir_path=model_dir)

def main():
    data_dir_path = './data'

    # fit the data and save model into model_dir_path
    if DO_TRAINING:

        ## Baseline model
        startModel(EventEmbLSTMConv1DNet(sequence_length = P, alphabet_size = A), './models/baseline', data_dir_path)

        ## Interval model
        startModel(IntervalLSTMConv1DNet(sequence_length = P), './models/interval', data_dir_path)

        ## Emb + Interval model
        startModel(EmbConcIntLSTMConv1DNet(sequence_length = P, alphabet_size = A), './models/emb_int_simple', data_dir_path)

        ## 2 RNN model
        startModel(EvtInt2RnnNet(sequence_length = P, alphabet_size = A), './models/emp_int_2rnn', data_dir_path)

        ## 1 RNN model
        startModel(EvtInt1RnnNet(sequence_length = P, alphabet_size = A), './models/emp_int_1rnn', data_dir_path)

        ## Dilated sequences with linear spaces
        startModel(LinearEmptyEvtsBaseline(sequence_length = int(P * 1.5), alphabet_size = A), './models/linear_empty_evts', data_dir_path)

        ## Dilated sequences with non-linear spaces
        startModel(NonLinearEmptyEvtsBaseline(sequence_length = int(P * 1.5), alphabet_size = A), './models/non_linear_empty_evts', data_dir_path)

        ## Intensity model
        startModel(IntensityEvt2RnnNet(sequence_length = P, alphabet_size = A), './models/intensity_event', data_dir_path)

if __name__ == '__main__':
    main()