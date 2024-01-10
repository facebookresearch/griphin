#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from setuptools import setup

from torch.utils import cpp_extension


setup(name='graph_engine',
      ext_modules=[cpp_extension.CppExtension('graph_engine',
                                              ['engine/bindings.cpp'],
                                              extra_compile_args=['-fopenmp',
                                                                  '-fvisibility=hidden',
                                                                  '-O3'],
                                              )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
