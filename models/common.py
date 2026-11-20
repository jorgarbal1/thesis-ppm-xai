import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Embedding, Input, Layer
from keras.models import Model
from keras.saving import register_keras_serializable


@register_keras_serializable(package = "v3lstm")
def f1(y_true, y_pred):
    """Batch-wise F1 score."""
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def embed_dim_for(n_categories):
    """Embedding size for a categorical variable with this many categories."""
    return min(600, round(1.6 * n_categories ** 0.56))


@register_keras_serializable(package = "v3lstm")
def _gather_fn(t, columns):
    """Picks the given columns from the last axis of t."""
    return tf.gather(t, columns, axis = -1)


@register_keras_serializable(package = "v3lstm")
class CategoricalEmbedding(Layer):
    """Embeds one categorical variable, zeroing out the embedding at padding positions."""
    def __init__(self, col_index, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.col_index = col_index
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = Embedding(vocab_size, embed_dim, name = (kwargs.get("name") or "cat") + "_table")

    def build(self, input_shape):
        self.embedding.build(input_shape[:-1])
        super().build(input_shape)

    def call(self, inputs):
        idx_int = tf.cast(inputs[:, :, self.col_index], "int32")
        emb = self.embedding(idx_int)
        mask = tf.cast(tf.not_equal(idx_int, 0), "float32")[:, :, None]
        return emb * mask

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.embed_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({"col_index": self.col_index, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return config


def build_embed_model(model):
    """Submodel: raw prefix -> concatenated embeddings + numericals."""
    return Model(model.input, model.get_layer("concat_features").output)


def build_explain_model(model):
    """Submodel: concatenated embeddings -> model's prediction."""
    concat_layer = model.get_layer("concat_features")
    new_input = Input(shape = concat_layer.output.shape[1:])
    x = new_input
    past_concat = False
    for layer in model.layers:
        if layer is concat_layer:
            past_concat = True
            continue
        if past_concat:
            x = layer(x)
    return Model(new_input, x)


def build_logits_model(model):
    """Submodel: concatenated embeddings -> pre-softmax logits (classification only)."""
    concat_layer = model.get_layer("concat_features")
    try:
        logits_layer = model.get_layer("logits")
    except ValueError:
        logits_layer = model.layers[-2]
        if not isinstance(logits_layer, Dense):
            raise ValueError(f"no se encontro la capa de logits: se esperaba Dense antes de la activacion "
                             f"final, salio {type(logits_layer).__name__}")
    new_input = Input(shape = concat_layer.output.shape[1:])
    x = new_input
    past_concat = False
    for layer in model.layers:
        if layer is concat_layer:
            past_concat = True
            continue
        if past_concat:
            x = layer(x)
            if layer is logits_layer:
                break
    return Model(new_input, x)


def build_raw_logits_model(model):
    """Submodel: raw prefix -> pre-softmax logits (classification only)."""
    embed_model = build_embed_model(model)
    logits_model = build_logits_model(model)
    return Model(embed_model.input, logits_model(embed_model.output))


def train_model(model, X_train, y_train, num_epochs=500, batch_size=32, checkpoint_path=None, initial_best=np.inf, verbose=1,
                 early_stopping_patience=42, reduce_lr_patience=10):
    """Trains with early stopping and reduce-lr-on-plateau, saving the best epoch to checkpoint_path."""
    callbacks = [
        EarlyStopping(patience=early_stopping_patience),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=reduce_lr_patience, verbose=0, mode="auto",
                          min_delta=0.0001, cooldown=0, min_lr=0),
    ]
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_loss", verbose=0, save_best_only=True,
                                     mode="auto", save_weights_only=False)
        checkpoint.best = initial_best
        callbacks.append(checkpoint)
    return model.fit(X_train, y_train, validation_split=0.2, callbacks=callbacks,
                     epochs=num_epochs, batch_size=batch_size, verbose=verbose)
