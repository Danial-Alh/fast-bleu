from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

ext_modules = [
    Extension(
        name="bleu",
        sources=['./cython_sources/bleu_cy.pyx'],
        extra_compile_args=['-fopenmp', '-std=c++11'],
        extra_link_args=['-fopenmp', '-std=c++11'],
        include_dirs=['./cpp_sources/headers/', './cpp_sources/sources/'],
    ),
    Extension(
        name="self_bleu",
        sources=['./cython_sources/self_bleu_cy.pyx'],
        extra_compile_args=['-fopenmp', '-std=c++11'],
        extra_link_args=['-fopenmp', '-std=c++11'],
        include_dirs=['./cpp_sources/headers/', './cpp_sources/sources/'],
    )
]

setup(
    name='cy_bleu_setup',
    ext_modules=cythonize(ext_modules, build_dir='./build'),
    requires=['Cython']
)
