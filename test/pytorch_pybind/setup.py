#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

# import sys
# from pybind11 import get_cmake_dir
# from pybind11.setup_helpers import Pybind11Extension, build_ext
# from setuptools import setup
#
# __version__ = "0.0.1"
#
# ext_modules = [
#     Pybind11Extension("python_example",
#                       ["example.cpp"],
#                       # Example: passing in the version to the compiled code
#                       define_macros=[('VERSION_INFO', __version__)],
#                       ),
# ]
#
# setup(
#     name="python_example",
#     version=__version__,
#     description="A test project using pybind11",
#     long_description="",
#     ext_modules=ext_modules,
#     extras_require={"test": "pytest"},
#     # Currently, build_ext only provides an optional "highest supported C++
#     # level" feature, but in the future it may provide more features.
#     cmdclass={"build_ext": build_ext},
#     zip_safe=False,
# )

from setuptools import setup, Extension

from torch.utils import cpp_extension
"""
    CppExtension is a convenience wrapper around setuptools.Extension that passes
    the correct include paths and sets the language of the extension to C++.
    The equivalent vanilla setuptools code would simply be:
    
    Extension(
       name='lltm_cpp',
       sources=['lltm.cpp'],
       include_dirs=cpp_extension.include_paths(),
       language='c++')
"""

setup(name='python_example',
      ext_modules=[cpp_extension.CppExtension('python_example',
                                              ['example.cpp'],
                                              # extra_compile_args=['-fopenmp'],
                                              )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})




