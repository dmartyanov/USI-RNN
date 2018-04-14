import os

from library.models import EventEmbLSTMConv1DNet

DO_TRAINING = True

def main():
    data_dir_path = './data'
    model_dir_path = './models/baseline'
    

    eventEmbRNNNet = EventEmbLSTMConv1DNet()

    # fit the data and save model into model_dir_path
    if DO_TRAINING:
        eventEmbRNNNet.fit(data_dir_path, model_dir_path=model_dir_path)


if __name__ == '__main__':
    main()