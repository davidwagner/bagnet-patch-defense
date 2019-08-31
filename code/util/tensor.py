## Copyright (C) 2017, 2019 Michael McCoyd <mmccoyd@cs.berkeley.edu>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in the root code directory.

"""
Functions that help in using tensorflow.
"""

import wrapt  # Better decorator utils than functools

def _d_print(inputs, name: str = 'Foo'):
  """Print shape of inputs, which is tensor or list of tensors."""
  is_list = isinstance(inputs, (list, tuple))
  print('{}:   ({})'.format(name, 'List' if is_list else 'Single'))
  if not is_list:
    print('   ', inputs.dtype, inputs.shape, inputs.name, flush=True)
  else:
    for ten in inputs:
      print('   ', ten.dtype, ten.shape, ten.name, flush=True)
  print('', flush=True)
  return inputs

def debug_build(enabled: bool):
  """Decorator logs Layer.build() input shape. `@debug_build(DEBUG)`"""
  @wrapt.decorator(enabled=enabled)
  def wrapper(wrapped, instance, args, kwargs):
    print('{} build() (in {}) input_shape {}'.format(
      instance.__class__.__name__, instance.__class__.__module__,
      *args), flush=True)
    return wrapped(*args, **kwargs)
  return wrapper

def debug_call(enabled: bool):
  """Decorator logs Layer.call() I/O shapes. `@debug_call(DEBUG)`"""
  @wrapt.decorator(enabled=enabled)
  def wrapper(wrapped, instance, args, kwargs):
    _d_print(*args, '{} call() (in {}) inputs'.format(
      instance.__class__.__name__, instance.__class__.__module__,))
    ret = wrapped(*args, **kwargs)
    _d_print(ret, '{} call() outputs'.format(instance.__class__.__name__))
    return ret
  return wrapper

def debug_compute(enabled: bool):
  """Decorator logs compute_output_shape output. `@debug_compute(DEBUG)`"""
  @wrapt.decorator(enabled=enabled)
  def wrapper(wrapped, instance, args, kwargs):
    print('{} compute_output_shape() (in {}) input  {}'.format(
      instance.__class__.__name__, instance.__class__.__module__,
      *args), flush=True)
    ret = wrapped(*args, **kwargs)
    print('{} compute_output_shape() output {}'.format(
      instance.__class__.__name__, ret), flush=True)
    return ret
  return wrapper
