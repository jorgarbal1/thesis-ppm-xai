"""Transformer of v3-transformer: ProcessTransformer's attention backbone (TransformerBlock + learned positional
encoding), adapted to accept every categorical/numerical variable the LSTM of v3-LSTM accepts, with the same
data conventions (see multi_loader.py/multi_processor.py, imported from v3-LSTM, not duplicated here) and the
same explainability invariants -- so that comparing explanation stability between the two architectures isn't
confounded by differences in data handling or in what "padding" means to each one.

Not a reimplementation of SuTraN or of the multimodal extension of ProcessTransformer: those solve a different
problem (best possible suffix/next-step predictor) and make choices -- a Dense projection after concatenating
embeddings, late fusion of numericals, GlobalAveragePooling1D, no padding mask -- that would introduce exactly
the kind of asymmetry between architectures this comparison needs to avoid. See the plan this was built from
for the full reasoning; the short version of each deviation from the original ProcessTransformer
(v3-transformer/processtransformer/processtransformer/models/transformer.py) is in this file's docstrings.

embed_dim_for, CategoricalEmbedding, build_embed_model, build_explain_model, build_logits_model,
build_raw_logits_model, train_model and f1 are the LSTM's own (v3-LSTM/multi_model.py), imported directly, not
copied: any future fix to them (or to the embed_model/explain_model cut convention) applies to both
architectures at once. Only the attention-specific layers and build_model are new here.
"""
import re

import tensorflow as tf
from keras.layers import (Activation, Concatenate, Dense, Dropout, Embedding, Input, Lambda, Layer,
                           LayerNormalization, MultiHeadAttention)
from keras.models import Model
from keras.saving import register_keras_serializable

# Both architectures now live in this same repo as proper sibling packages (models.lstm.model /
# models.transformer.model), so the module-name collision the old v3-transformer/v3-LSTM layout had (both
# literally named multi_model.py, forcing the importlib-by-file-path workaround) no longer exists -- a plain
# import is enough.
from models.lstm import model as _lstm_model

f1 = _lstm_model.f1
embed_dim_for = _lstm_model.embed_dim_for
CategoricalEmbedding = _lstm_model.CategoricalEmbedding
_gather_fn = _lstm_model._gather_fn
build_embed_model = _lstm_model.build_embed_model
build_explain_model = _lstm_model.build_explain_model
build_logits_model = _lstm_model.build_logits_model
build_raw_logits_model = _lstm_model.build_raw_logits_model
train_model = _lstm_model.train_model


@register_keras_serializable(package = "v3transformer")
class PositionalEncoding(Layer):
    """Learned positional embedding (same scheme as the original ProcessTransformer's TokenAndPositionEmbedding),
    added to the concatenated embeddings. Unlike the original, this sits *after* the embed_model/explain_model
    cut (build_explain_model's input is the output of "concat_features", before this layer runs) -- positional
    encoding is a deterministic function of position, not of the value being explained, so IG's baseline can
    stay a pure content embedding (built exactly like the LSTM's) without having to account for position.

    Padding rows are exactly the zero vector coming in (CategoricalEmbedding forces it); summing a nonzero
    positional vector into them would silently turn "absence" into "absence + a position", so they're forced
    back to zero right after the sum. The row is identified as padding by being the exact zero vector itself --
    not a separate length input, not Keras's automatic mask propagation -- so this stays a function of its own
    single input tensor (required for TimeSHAP's and the IG wrappers' f(x) with one argument)."""
    def __init__(self, maxlen, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_emb = Embedding(maxlen, embed_dim, name = (kwargs.get("name") or "pos") + "_table")

    def build(self, input_shape):
        self.pos_emb.build((None, self.maxlen))
        super().build(input_shape)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = self.pos_emb(tf.range(length))                                  # (max_len, embed_dim)
        is_real = tf.reduce_any(tf.not_equal(inputs, 0.0), axis = -1, keepdims = True)  # (batch, max_len, 1)
        return (inputs + positions) * tf.cast(is_real, inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "embed_dim": self.embed_dim})
        return config


@register_keras_serializable(package = "v3transformer")
class MaskedTransformerBlock(Layer):
    """The original ProcessTransformer's TransformerBlock (MultiHeadAttention + FFN, residual + layernorm around
    each), plus a padding mask in the attention -- the original has none, for any purpose (checked: no "mask" of
    any kind anywhere in that package). That isn't a modernization, it's required by the same zero=absence
    invariant the embeddings already enforce: softmax normalizes over every included key regardless of its
    content, and MultiHeadAttention's internal Q/K/V projections have a bias term, so an all-zero padding row
    does NOT produce an all-zero (ignorable) key/value without an explicit mask -- it would still take a share
    of the attention and still contribute a nonzero, bias-driven value. No causal mask: every prefix is predicted
    once, from the whole prefix (see multi_loader.py), so there is no "future" to hide within it.

    The mask (which rows are real vs. padding) is derived from this layer's own input -- a row is padding iff it
    is exactly the zero vector -- for the same single-tensor-input reason as PositionalEncoding. The output rows
    that were padding on the way in are forced back to exactly zero before returning: attention still computes
    *something* for a padding query (attending over the real keys), so without this, padding would stop being
    the zero vector after one block, and a second stacked block would no longer be able to tell it apart from a
    real event. Only extract_last_token's output is ever used downstream (padding is never the last position,
    see multi_loader.py's left-padding), but this keeps the invariant correct for any number of stacked blocks."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.num_heads, self.ff_dim, self.rate = embed_dim, num_heads, ff_dim, rate
        self.att = MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation = "relu"), Dense(embed_dim)])
        self.layernorm_a = LayerNormalization(epsilon = 1e-6)
        self.layernorm_b = LayerNormalization(epsilon = 1e-6)
        self.dropout_a = Dropout(rate)
        self.dropout_b = Dropout(rate)

    def build(self, input_shape):
        # same reason as CategoricalEmbedding.build above: sublayers created in __init__ build lazily on their
        # first real call, which happens during training but never during weight loading (Keras reconstructs
        # the layer graph and tries to assign saved weights before any such call happens) -- without building
        # them here explicitly, load_model() fails with "layer was never built" for att/ffn/layernorm_a/b.
        self.att.build(input_shape, input_shape)   # query_shape, value_shape (key defaults to value, as in call)
        self.ffn.build(input_shape)
        self.layernorm_a.build(input_shape)
        self.layernorm_b.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training = None):
        is_real = tf.reduce_any(tf.not_equal(inputs, 0.0), axis = -1)         # (batch, max_len): True = real event
        real_mask = tf.cast(is_real, inputs.dtype)[:, :, None]                 # (batch, max_len, 1)
        attention_mask = is_real[:, tf.newaxis, :]                             # (batch, 1, max_len): broadcasts over queries

        attn_output = self.att(inputs, inputs, attention_mask = attention_mask)
        attn_output = self.dropout_a(attn_output, training = training)
        out_a = self.layernorm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output, training = training)
        out_b = self.layernorm_b(out_a + ffn_output)
        return out_b * real_mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                       "ff_dim": self.ff_dim, "rate": self.rate})
        return config


@register_keras_serializable(package = "v3transformer")
class ExtractLastToken(Layer):
    """Picks the representation of the last position. Valid because padding is on the left (multi_loader.py:
    `X[r, max_len - matrix.shape[0]:, :] = matrix`), so position -1 is always the last real event of the prefix.

    Analogous to the LSTM's final hidden state, not a simplification of it: the LSTM stack has
    return_sequences=True on every layer but the last, so what reaches its head is the state after recurrently
    processing the whole prefix, not just the last event's raw input. Here, after MaskedTransformerBlock's
    (unmasked-by-causality) attention, position -1's representation has likewise already mixed in every earlier
    position through the attention weights. Both are a single summary vector that has integrated the whole
    prefix, just via a different mixing mechanism (attention weights vs. recurrent gates) -- which is what makes
    this the right analogue for comparing the two architectures, not GlobalAveragePooling1D (fixed, unlearned
    weight per position, as the original ProcessTransformer uses)."""
    def call(self, inputs):
        return inputs[:, -1, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def build_model(max_len, vocab_sizes, categorical_columns, numerical_columns,
                 n_blocks = 1, num_heads = 4, ff_dim = 64, dropout_rate = 0.1,
                 output_dim = 1, task = "regression", event_level = True):
    """Same role as v3-LSTM's build_model (and the same signature, as far as the architectures allow): builds the
    full model from the loader's per-variable vocab/column info, with a Concatenate(name="concat_features") cut
    point and, for classification, a separate Dense(name="logits") before the final Activation -- both required,
    unmodified, by build_embed_model/build_explain_model/build_logits_model/build_raw_logits_model.

    n_blocks, num_heads, ff_dim: same meaning and same defaults (1 block, 4 heads, ff_dim=64) as the original
    ProcessTransformer, which only ever had one categorical variable. Starting point, not a ceiling: with many
    more variables than the original, watch for signs that 1 block is a capacity bottleneck (e.g. underfitting
    relative to the LSTM) and stack more if so.

    dropout_rate: passed through to every MaskedTransformerBlock's two Dropout layers (same default, 0.1, as
    the original ProcessTransformer's TransformerBlock). Raising ff_dim adds capacity without adding
    regularization to match it -- if that shows up as overfitting (train accuracy/loss still improving while
    val_loss gets worse epoch after epoch, as observed when ff_dim was raised from 64 to 256 on next_activity),
    raising dropout_rate alongside it is the first lever to try, before changing n_blocks again.

    No Dense projection is applied after "concat_features" (unlike SuTraN's nn.Linear(dim_init_prefix, d_model)
    and the multimodal ProcessTransformer's emb_projection): the concatenated width becomes embed_dim directly,
    so the Transformer's input representation has exactly the same capacity as the LSTM's -- which feeds its
    concat_features straight into the LSTM stack, with no projection either. Deliberate, so that any difference
    found in explanation stability is attributable to the sequential architecture, not to embedding capacity.
    """
    width = max(list(categorical_columns.values()) + numerical_columns) + 1   # columns in the raw (max_len, F) input
    inputs = Input(shape = (max_len, width), name = "prefix")

    embeds = [CategoricalEmbedding(col, vocab_sizes[name], embed_dim_for(vocab_sizes[name] - 2),
                                    name = f"cat_embed_{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}")(inputs)
              for name, col in categorical_columns.items()]
    numeric = Lambda(_gather_fn, arguments = {"columns": numerical_columns},
                      output_shape = (max_len, len(numerical_columns)),
                      name = "numerical_slice")(inputs)

    x = Concatenate(axis = -1, name = "concat_features")(embeds + [numeric])
    embed_dim = x.shape[-1]   # sum of the per-variable embedding sizes + numericals: no bottleneck projection

    x = PositionalEncoding(max_len, embed_dim, name = "positional_encoding")(x)
    for i in range(n_blocks):
        x = MaskedTransformerBlock(embed_dim, num_heads, ff_dim, rate = dropout_rate, name = f"transformer_block_{i}")(x)
    x = ExtractLastToken(name = "extract_last_token")(x)

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
