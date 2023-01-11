from setuptools import setup

from torch.utils import cpp_extension


setup(name='graph_engine',
      ext_modules=[cpp_extension.CppExtension('graph_engine',
                                              ['engine/bindings.cpp'],
                                              extra_compile_args=['-fopenmp'],
                                              )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
