from tfx import v1 as tfx
import os
from absl import logging
import urllib.request
import tempfile

# ------------------ Set up variables -----------------------------
# Define the name of the pipeline.
PIPELINE_NAME = "penguin-simple"
# Output directory to store artifacts generated from the pipeline.
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
# Path to a SQLite DB file to use as an ML-MD (Machine Learning Metadata) storage.
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')
# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)
# Set default logging level.
logging.set_verbosity(logging.INFO)

# -------------------------- Prepare example data -------------------------
# Create a temporary directory to store example data.
DATA_ROOT = tempfile.mkdtemp(prefix='tfx-data')
# URL to the CSV data file.
_data_url = ('https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/penguin/data/labelled'
             '/penguins_processed.csv')
# Download the data file and save it to the temporary directory.
_data_filepath = os.path.join(DATA_ROOT, "data.csv")
urllib.request.urlretrieve(_data_url, _data_filepath)


# -------------------------- Write a pipeline definition -------------------------

# Define a CSVExampleGen component to ingest and process the example data.
example_gen = tfx.components.CsvExampleGen(input_base=DATA_ROOT)

# Define a Trainer component that uses a user-provided Python function to train a model.
trainer = tfx.components.Trainer(
    module_file='trainer.py',  # Path to the Python module containing the training code.
    examples=example_gen.outputs['examples'],
    train_args=tfx.proto.TrainArgs(num_steps=100),  # Training configuration.
    eval_args=tfx.proto.EvalArgs(num_steps=5))  # Evaluation configuration.

# Define a Pusher component to push the trained model to a filesystem destination.
pusher = tfx.components.Pusher(
    model=trainer.outputs['model'],
    push_destination=tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(
            base_directory=SERVING_MODEL_DIR)))  # Directory to export the model.

# Define the TFX pipeline with the specified components.
pipeline = tfx.dsl.Pipeline(
    pipeline_name=PIPELINE_NAME,
    pipeline_root=PIPELINE_ROOT,
    metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH),
    components=[example_gen, trainer, pusher])

# Run the TFX pipeline using the LocalDagRunner for local execution.
tfx.orchestration.LocalDagRunner().run(pipeline)
