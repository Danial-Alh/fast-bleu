from glob import glob

import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension


class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        super().get_ext_filename(ext_name)
        return ext_name + '.so'
    # pass


include_dirs = ['fast_bleu/cpp_sources/headers/']
setup = setuptools.setup(
    name='fast_bleu',
    ext_modules=[
        Extension(
            name="fast_bleu.__fast_bleu_module",
            sources=glob('fast_bleu/cpp_sources/sources/*.cpp'),
            extra_compile_args=['-fopenmp', '-std=c++11'],
            extra_link_args=['-fopenmp', '-std=c++11'],
            include_dirs=include_dirs,
        ), ],
    cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
    packages=['fast_bleu'],
)
