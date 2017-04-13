import keras
import tensorflow
from keras import backend as K
from tensorflow.contrib.session_bundle import exporter
from keras.models import model_from_config, Sequential

print("Loading model for exporting to Protocol Buffer format...")
model_path = "simple.h5"
model = keras.models.load_model(model_path)

K.set_learning_phase(0)  # all new operations will be in test mode from now on
sess = K.get_session()

# serialize the model and get its weights, for quick re-building
config = model.get_config()
weights = model.get_weights()

# re-build a model where the learning phase is now hard-coded to 0
new_model = Sequential.from_config(config)
new_model.set_weights(weights)

export_path = "simple.pb"  # where to save the exported graph
export_version = 1  # version number (integer)

saver = tensorflow.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor=model.input, scores_tensor=model.output)
model_exporter.init(sess.graph.as_graph_def(), default_graph_signature=signature)
model_exporter.export(export_path, tensorflow.constant(export_version), sess)
