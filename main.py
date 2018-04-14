import os
import shutil

from library.models import EventEmbLSTMConv1DNet

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

if __name__ == '__main__':
    main()