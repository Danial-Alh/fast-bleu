from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

ext_modules = [
    Extension(
        name="bleu",
        sources=['./cython_sources/bleu_cy.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=['./cython_sources/', './cpp_sources/'],
    )
]

setup(
    name='cy_bleu_setup',
    ext_modules=cythonize(ext_modules, build_dir='./build'),
    requires=['Cython']
)
