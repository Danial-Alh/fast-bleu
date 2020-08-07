from glob import glob

import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension


class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        super().get_ext_filename(ext_name)
        return ext_name + '.so'


with open("README.md", "r") as fh:
    long_description = fh.read()

include_dirs = ['fast_bleu/cpp_sources/headers/']
setup = setuptools.setup(
    name='fast-bleu',
    version="0.0.85",
    author="Danial Alihosseini",
    author_email="danial.alihosseini@gmail.com",
    description="A fast multithreaded C++ implementation of nltk BLEU.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Danial-Alh/fast-bleu",
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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 3 - Alpha",
    ],
    python_requires='>=3.3',
    install_requires=[],
    obsoletes=["FastBLEU"],
    platforms=['POSIX :: Linux'],
    license='OSI Approved :: MIT License'
)
