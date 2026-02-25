import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.public import tfxio

_FEATURE_KEYS = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
_LABEL_KEY = 'species'

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10


def _input_fn(file_pattern, data_accessor, schema, batch_size):
    """Generates features and label for training.

      Args:
        file_pattern: List of paths or patterns of input tfrecord files.
        data_accessor: DataAccessor for converting input to RecordBatch.
        schema: schema of the input data.
        batch_size: representing the number of consecutive elements of returned
          dataset to combine in a single batch

      Returns:
        A dataset that contains (features, indices) tuple where features is a
          dictionary of Tensors, and indices is a single Tensor of label indices.
      """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key=_LABEL_KEY),
        schema=schema).repeat()


def _build_keras_model():
    """Creates a DNN Keras model for classifying penguin data.

     Returns:
       A Keras Model.
     """
    inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
    d = keras.layers.concatenate(inputs)
    d = keras.layers.Dense(8, activation='relu')(d)
    outputs = keras.layers.Dense(3)(d)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def run_fn(fn_args):
    """Train the model based on given args.

      Args:
        fn_args: Holds args used to train the model as name/value pairs.
      """
    schema = schema_utils.schema_from_feature_spec({
        **{feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for feature in _FEATURE_KEYS},
        _LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
    })

    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, schema, _TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema, _EVAL_BATCH_SIZE)

    model = _build_keras_model()
    model.fit(train_dataset, steps_per_epoch=fn_args.train_steps, validation_data=eval_dataset,
              validation_steps=fn_args.eval_steps)

    model.save(fn_args.serving_model_dir, save_format='tf')
