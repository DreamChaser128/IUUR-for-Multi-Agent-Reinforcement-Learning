import collections
import numpy as np
import tensorflow as tf

def is_placeholder(x):
    return type(x) is tf.Tensor and len(x.op.inputs) == 0

# ================================================================
# Inputs
# ================================================================


class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplemented()

    def make_feed_dict(data):
        """Given data input it to the placeholder(s)."""
        raise NotImplemented()


class PlacholderTfInput(TfInput):
    def __init__(self, placeholder):
        """Wrapper for regular tensorflow placeholder."""
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}

def get_session():
    """Returns recently made Tensorflow session"""
    return tf.get_default_session()

# ================================================================
# Theano-like Function
# ================================================================


def function(inputs, outputs, updates=None, givens=None,type_f=None):
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, type_f,givens=givens)
    #     return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates,type_f, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates,type_f, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates,type_f, givens, check_nan=False):
        self.type_f=type_f
        for inpt in inputs:
            if not issubclass(type(inpt), TfInput) and not issubclass(type(inpt), int):
                assert len(inpt.op.inputs) == 0, "inputs should all be placeholders of rl_algs.common.TfInput"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = [outputs] + [self.update_group]

        self.givens = {} if givens is None else givens
        self.check_nan = check_nan

    def _feed_input(self, feed_dict, inpt, value):
        if issubclass(type(inpt), TfInput):
            feed_dict.update(inpt.make_feed_dict(value))
        elif is_placeholder(inpt):
            feed_dict[inpt] = value

    def __call__(self, *args, **kwargs):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):#--
            self._feed_input(feed_dict, inpt, value)
        # Update the kwargs
        kwargs_passed_inpt_names = set()
        for inpt in self.inputs[len(args):]:#---
            inpt_name = inpt.name.split(':')[0]
            inpt_name = inpt_name.split('/')[-1]
            assert inpt_name not in kwargs_passed_inpt_names, \
                "this function has two arguments with the same name \"{}\", so kwargs cannot be used.".format(inpt_name)
            if inpt_name in kwargs:
                kwargs_passed_inpt_names.add(inpt_name)
                self._feed_input(feed_dict, inpt, kwargs.pop(inpt_name))
            else:
                assert inpt in self.givens, "Missing argument " + inpt_name
        assert len(kwargs) == 0, "Function got extra arguments " + str(list(kwargs.keys()))
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("Nan detected")
        return results
