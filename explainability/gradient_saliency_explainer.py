import numpy as np
import tensorflow as tf

from preprocessing.processor import Task


def explain_gradient_saliency(model, model_module, task, prefix, length, column_blocks, feature_names,
                               class_idx = None, y_minmax = None):
    """Input x Gradient saliency: the gradient multiplied by the input, as a signed sum and an absolute-value
    sum per variable. For regression explains the model output transformed to days; for classification,
    the logits.
    Args:
        model: the trained model to explain
        model_module: the model's own module (models/lstm/model.py or models/transformer/model.py)
        task: Task.NEXT_ACTIVITY or Task.REMAINING_TIME
        prefix: (max_len, F) for one prefix, or (B, max_len, F) for a batch
        length: real (unpadded) length of the prefix(es) being explained
        column_blocks: dict: variable name -> its column slice in embedding space
        feature_names: variable names, in the order the output matrices should have
        class_idx: classification only: fixed class for the whole batch, or None for each row's own predicted
            class
        y_minmax: regression only: (y_min, y_max) in seconds the target was scaled with

    Returns (grad_x_input_signed, grad_x_input_abs). Single-prefix mode gives a (length, F) matrix each;
    batch mode gives a (B, length, F) matrix each."""
    batched = prefix.ndim == 3
    prefix_batch = prefix if batched else prefix[None, ...]
    nb_samples = prefix_batch.shape[0]
    max_len = prefix_batch.shape[1]
    embed_model = model_module.build_embed_model(model)

    z_x = embed_model.predict(prefix_batch, verbose = 0)
    xs = tf.convert_to_tensor(z_x, dtype = tf.float32)

    if task == Task.REMAINING_TIME:
        target_model = model_module.build_explain_model(model)
        with tf.GradientTape() as tape:
            tape.watch(xs)
            out = target_model(xs, training = False)[:, 0:1]
        grad = tape.gradient(out, xs).numpy()
    else:
        target_model = model_module.build_logits_model(model)
        logits_x = target_model.predict(z_x, verbose = 0)
        if class_idx is None:
            target_arg = np.argmax(logits_x, axis = 1)
        else:
            target_arg = np.full(nb_samples, class_idx, dtype = int)
        with tf.GradientTape() as tape:
            tape.watch(xs)
            out = target_model(xs, training = False)
            picked = tf.gather(out, target_arg, axis = 1, batch_dims = 1)
        grad = tape.gradient(picked, xs).numpy()

    grad_x_input = grad * z_x

    start = max_len - length

    def collapse_signed(attr):
        return np.stack([attr[:, start:, column_blocks[name]].sum(axis = -1) for name in feature_names], axis = -1)

    def collapse_abs(attr):
        return np.stack([np.abs(attr[:, start:, column_blocks[name]]).sum(axis = -1) for name in feature_names], axis = -1)

    grad_x_input_signed = collapse_signed(grad_x_input)
    grad_x_input_abs = collapse_abs(grad_x_input)

    if task == Task.REMAINING_TIME:
        y_min, y_max = y_minmax
        scale = (y_max - y_min) / 86400.0
        grad_x_input_signed = grad_x_input_signed * scale
        grad_x_input_abs = grad_x_input_abs * scale

    if not batched:
        return grad_x_input_signed[0], grad_x_input_abs[0]
    return grad_x_input_signed, grad_x_input_abs
