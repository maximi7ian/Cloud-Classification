
import tensorflow as tf
from tensorflow.keras import layers
import argparse
import os
import numpy as np
import json
from sklearn.metrics import accuracy_score

from tensorflow.keras.optimizers import Adam

def cloudDetectorModel(lr):    

    model = tf.keras.Sequential()
    # Augmentation
    model.add(layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(125, 125, 3)))
    model.add(layers.experimental.preprocessing.RandomRotation(0.1))
    model.add(layers.experimental.preprocessing.RandomZoom(0.1))

    model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(125, 125, 3)))
    model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    return model



def main(x_train, y_train, x_test, y_test):


    tf_writer = tf.summary.create_file_writer(args.tb_callback)
    tf_writer.set_as_default()

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=args.tb_callback, update_freq='epoch', histogram_freq=1)


    mdl = cloudDetectorModel(args.learning_rate)

    mdl.fit(
        x_train, y_train, 
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,      
        callbacks=[tb_callback]
    )

    tf_writer.flush()

    if args.current_host == args.hosts[0]:
        mdl.save(os.path.join(args.sm_model_dir, '01'))
  





def _load_training_data(base_dir):
    x_train = np.load(os.path.join(base_dir, 'train_data.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(base_dir, 'train_labels.npy'), allow_pickle=True)
    return x_train, y_train


def _load_testing_data(base_dir):
    x_test = np.load(os.path.join(base_dir, 'test_data.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(base_dir, 'test_labels.npy'), allow_pickle=True)
    return x_test, y_test




def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--tb_callback', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()



if __name__ == '__main__':
    args, unknown = _parse_args()

    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.train)
    
    main(train_data, train_labels, eval_data, eval_labels)

    