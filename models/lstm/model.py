"""LSTM of v3-LSTM: same recurrent architecture as v2-LSTM (and ultimately LSTM_Generic_explainable), but the
input is no longer one-hot. Every categorical variable is label-encoded (see multi_loader.py) and goes through
its own Embedding layer; the embeddings and the scaled numerical columns are concatenated, and that is what feeds
the stack of LSTM layers. The LSTM stack itself, the heads and train_model are unchanged from v2-LSTM.

Why: Integrated Gradients (and GradientExplainer) need a continuous input to interpolate against a baseline.
One-hot technically works (it's differentiable), but the LSTM's own first weight matrix already acts as an
implicit embedding (one_hot @ W selects a row of W); this version makes that embedding explicit, small, and
exposes it directly, so IG can be computed on it instead of relying on that implicit equivalence.

The embedding of the padding index (0) is forced to the zero vector regardless of what that row of the
Embedding table learns (see _masked_embedding): padding stays at exactly zero everywhere in the model, the same
convention used everywhere else (padding positions of X, the Kernel's "always kept" padding, IG's zero baseline).
"""
import os
import re

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (LSTM, BatchNormalization, Dense, Activation, Input, Embedding,
                           Concatenate, Lambda, Layer)
from keras.models import Model
from keras.saving import register_keras_serializable


@register_keras_serializable(package = "v3lstm")
def f1(y_true, y_pred):
    """Batch-wise F1 for multi-label outputs (as in the original)."""
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
    """Embedding size rule of SuTraN (SuffixTransformerNetwork/SuTraN/SuTraN.py), itself the fastai tabular-learner
    heuristic: min(600, round(1.6 * n_categories**0.56)). `n_categories` excludes the padding and "unseen" slots."""
    return min(600, round(1.6 * n_categories ** 0.56))


@register_keras_serializable(package = "v3lstm")
def _gather_fn(t, columns):
    """Lambda `function` for the numerical columns (picked by index, via `arguments=`): a plain module-level
    function (no closure), registered so Keras can serialize/deserialize it by reference instead of pickling a
    Python lambda's bytecode (which Keras 3 refuses to load back by default, see safe_mode)."""
    return tf.gather(t, columns, axis = -1)


@register_keras_serializable(package = "v3lstm")
class CategoricalEmbedding(Layer):
    """One variable, one instance: picks its column out of the full (max_len, F) input, looks it up in its own
    Embedding table, and forces the embedding of the padding index (0) to the zero vector, regardless of what
    that row of the table learns. One class, reused once per categorical variable with its own col_index,
    vocab_size and embed_dim -- so model.summary() shows one row per variable instead of three."""
    def __init__(self, col_index, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.col_index = col_index
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = Embedding(vocab_size, embed_dim, name = (kwargs.get("name") or "cat") + "_table")

    def build(self, input_shape):
        # must build the Embedding sub-layer explicitly here (its weights, not just this layer's), otherwise
        # saving/loading a model with this layer fails: Keras doesn't know how to rebuild the child layer's state
        self.embedding.build(input_shape[:-1])   # the embedding's own input is (batch, max_len), one column of `inputs`
        super().build(input_shape)

    def call(self, inputs):
        idx_int = tf.cast(inputs[:, :, self.col_index], "int32")        # (batch, max_len)
        emb = self.embedding(idx_int)                                   # (batch, max_len, embed_dim)
        mask = tf.cast(tf.not_equal(idx_int, 0), "float32")[:, :, None]  # 0 at padding, 1 elsewhere
        return emb * mask

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.embed_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({"col_index": self.col_index, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return config


def build_model(max_len, vocab_sizes, categorical_columns, numerical_columns,
                 n_neurons = 100, n_layers = 8, output_dim = 1, task = "regression", event_level = True):
    """Same architecture as v2-LSTM's build_model (stacked LSTM + BatchNormalization, same heads), but the input
    is built from per-variable embeddings instead of one-hot.

    max_len, vocab_sizes, categorical_columns, numerical_columns: straight from the loader's returned dict
    (vocab_sizes, categorical_columns, numerical_columns), all fitted on train only.
    """
    width = max(list(categorical_columns.values()) + numerical_columns) + 1   # columns in the raw (max_len, F) input
    inputs = Input(shape = (max_len, width), name = "prefix")

    # nombre de capa de Keras: solo caracteres validos de scope (sin espacios ni otros simbolos), distinto del
    # nombre de la variable en si (que sigue siendo la clave real en categorical_columns/vocab_sizes)
    embeds = [CategoricalEmbedding(col, vocab_sizes[name], embed_dim_for(vocab_sizes[name] - 2),
                                    name = f"cat_embed_{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}")(inputs)
              for name, col in categorical_columns.items()]
    numeric = Lambda(_gather_fn, arguments = {"columns": numerical_columns},
                      output_shape = (max_len, len(numerical_columns)),
                      name = "numerical_slice")(inputs)

    x = Concatenate(axis = -1, name = "concat_features")(embeds + [numeric])

    if n_layers == 1:
        x = LSTM(n_neurons, implementation = 2, recurrent_dropout = 0.2)(x)
        x = BatchNormalization()(x)
    else:
        x = LSTM(n_neurons, implementation = 2, recurrent_dropout = 0.2, return_sequences = True)(x)
        x = BatchNormalization()(x)
        for _ in range(n_layers - 2):
            x = LSTM(n_neurons, implementation = 2, recurrent_dropout = 0.2, return_sequences = True)(x)
            x = BatchNormalization()(x)
        x = LSTM(n_neurons, implementation = 2, recurrent_dropout = 0.2)(x)
        x = BatchNormalization()(x)

    if task == "regression":
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(loss = "mae", optimizer = "Nadam", metrics = ["mean_squared_error", "mae"])
    else:
        x = Dense(output_dim, name = "logits")(x)
        if event_level:
            outputs = Activation("sigmoid")(x)
            model = Model(inputs, outputs)
            model.compile(loss = "binary_crossentropy", optimizer = "Nadam", metrics = ["accuracy", "binary_accuracy", f1])
        else:
            outputs = Activation("softmax")(x)
            model = Model(inputs, outputs)
            # one next activity per prefix: a single active output, so accuracy only (f1 is for multi-label outputs)
            model.compile(loss = "categorical_crossentropy", optimizer = "Nadam", metrics = ["accuracy"])
    return model


def build_embed_model(model):
    """Submodel: same input as `model` (the prefix, with integer-coded categoricals), output = the concatenated
    embeddings + numericals tensor fed to the first LSTM. No new layers, no training: reuses model's own graph."""
    return Model(model.input, model.get_layer("concat_features").output)


def build_explain_model(model):
    """Submodel for Integrated Gradients: input = the concatenated embeddings tensor itself (same shape as
    build_embed_model's output), output = model's prediction. Reuses the exact same LSTM/BatchNorm/head layer
    objects as `model` (no retraining, no weight copying); assumes -- true for this architecture -- that nothing
    branches after "concat_features", i.e. model.layers lists everything after it in execution order."""
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
    """Classification only: like build_explain_model, but stops at the pre-softmax logits (the
    Dense(output_dim, name="logits") layer), skipping the final Activation. Explaining logits instead of
    probabilities avoids softmax saturation -- the gradient of a softmax probability w.r.t. its own logit is
    p*(1-p), which goes to zero as p approaches 0 or 1 -- which otherwise starves Integrated Gradients/TimeSHAP
    of signal exactly where the predicted class is most confident (and is the likely cause of the poor
    TimeSHAP/IG agreement observed on next_activity when explaining the softmax probability directly)."""
    concat_layer = model.get_layer("concat_features")
    try:
        logits_layer = model.get_layer("logits")
    except ValueError:
        # modelo entrenado antes de nombrar la capa explicitamente: la logits es la que alimenta
        # directamente a la Activation final (Dense -> Activation es siempre el patron de la cabeza)
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
    """Classification only: raw prefix (label-encoded, same input as `model`) -> logits (pre-softmax), in a
    single Model. Composes build_embed_model + build_logits_model into one graph. This is what TimeSHAP should
    use as f(x): TimeSHAP perturbs/masks the raw label-encoded input directly (coalitions over the original
    columns), not the embedding space -- unlike IG, which interpolates in embedding space via build_embed_model/
    build_explain_model. Same underlying trained weights either way, no retraining."""
    embed_model = build_embed_model(model)
    logits_model = build_logits_model(model)
    return Model(embed_model.input, logits_model(embed_model.output))


def train_model(model, X_train, y_train, num_epochs=500, batch_size=32, checkpoint_path=None, initial_best=np.inf, verbose=1,
                 early_stopping_patience=42, reduce_lr_patience=10):
    """Same callbacks as the original (defaults unchanged: early_stopping_patience=42, reduce_lr_patience=10).

    early_stopping_patience / reduce_lr_patience: exposed (not hardcoded) because the two architectures can need
    different values here without that being an architecture-invariant decision -- e.g. a backbone whose val_loss
    plateaus and starts degrading within the first few epochs needs a much tighter reduce_lr_patience than 10 to
    react before the damage is done (ReduceLROnPlateau only fires after `reduce_lr_patience` epochs with no
    improvement, and never rewinds the weights -- it cannot undo overfitting that already happened while waiting).
    Named separately (not a single shared "patience") because they are two different callbacks with two different
    jobs: one stops training, the other only lowers the learning rate.

    If checkpoint_path is given, the best epoch (lowest val_loss) is saved there as a full model (.keras), so it can be
    loaded later with keras.models.load_model. initial_best is the best val_loss already reached in a previous run: the
    checkpoint is only overwritten by an epoch that beats it, so continuing a run never replaces a better model."""
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
