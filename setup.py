import os

from glob import glob

import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
from setuptools.command.install import install


class CleanCommand(setuptools.Command):
    """
    Our custom command to clean out junk files.
    """
    description = "Cleans out junk files we don't want in the repo"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cmd_list = dict(
            build="set -x; rm -rf ./build",
            dist="set -x; rm -rf ./dist",
            egg="set -x; rm -rf ./fast_bleu.egg-info",
            so="set -x; find . -name '*.so' -exec rm -rf {} \;",
            pyc="set -x; find . -name '*.pyc' -exec rm -rf {} \;"
        )
        for key, cmd in cmd_list.items():
            os.system(cmd)


class InstallCommand(install):
    user_options = install.user_options + [
        ('CC=', None, '<gcc binary path>'),
        ('CXX=', None, '<g++ binary path>'),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.CC = None
        self.CXX = None

    def finalize_options(self):
        print("Value of CC is", self.CC)
        print("Value of CXX is", self.CXX)
        install.finalize_options(self)

    def run(self):
        if self.CC is not None:
            os.environ["CC"] = self.CC
            print("CC changed to {}".format(self.CC))
        if self.CXX is not None:
            os.environ["CXX"] = self.CXX
            print("CXX changed to {}".format(self.CXX))
        install.run(self)


class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        super().get_ext_filename(ext_name)
        return ext_name + '.so'


with open("README.md", "r") as fh:
    long_description = fh.read()

setup = setuptools.setup(
    name='fast-bleu',
    version="0.0.90",
    author="Danial Alihosseini",
    author_email="danial.alihosseini@gmail.com",
    description="A fast multithreaded C++ implementation of nltk BLEU with python wrapper.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Danial-Alh/fast-bleu",
    ext_modules=[
        Extension(
            name="fast_bleu.__fast_bleu_module",
            sources=glob('fast_bleu/cpp_sources/sources/*.cpp'),
            extra_compile_args=['-fopenmp', '-std=c++11', '-Werror', '-pedantic-errors', '-Wall', '-Wextra'],
            extra_link_args=['-fopenmp', '-std=c++11', '-Werror', '-pedantic-errors', '-Wall', '-Wextra'],
            include_dirs=['fast_bleu/cpp_sources/headers/'],
        ), ],
    cmdclass={
        'install': InstallCommand,
        'build_ext': BuildExtWithoutPlatformSuffix,
        'clean': CleanCommand
    },
    packages=['fast_bleu'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Development Status :: 3 - Alpha",
    ],
    python_requires='>=3.3',
    install_requires=[],
    obsoletes=["FastBLEU"],
    platforms=['POSIX :: Linux', 'MacOS'],
    license='OSI Approved :: MIT License'
)
