import os
import shutil

from library.models import EventEmbLSTMConv1DNet
from library.models import IntervalLSTMConv1DNet
from library.models import EmbConcIntLSTMConv1DNet
from library.models import EvtInt2RnnNet
from library.models import EvtInt1RnnNet

DO_TRAINING = True

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
        startModel(EventEmbLSTMConv1DNet(), './models/baseline', data_dir_path)

        ## Interval model
        startModel(IntervalLSTMConv1DNet(), './models/interval', data_dir_path)

        ## Emb + Interval model
        startModel(EmbConcIntLSTMConv1DNet(), './models/emb_int_simple', data_dir_path)

        ## 2 RNN model
        startModel(EvtInt2RnnNet(), './models/emp_int_2rnn', data_dir_path)

        ## 1 RNN model
        startModel(EvtInt1RnnNet(), './models/emp_int_1rnn', data_dir_path)

if __name__ == '__main__':
    main()