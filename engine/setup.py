from setuptools import setup

from torch.utils import cpp_extension


setup(name='graph_engine',
      ext_modules=[cpp_extension.CppExtension('graph_engine',
                                              ['bindings.cpp'],
                                              extra_compile_args=['-fopenmp'],
                                              extra_cflags=['-O3']
                                              )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
