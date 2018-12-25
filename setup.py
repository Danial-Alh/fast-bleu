from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

ext_modules = [
    Extension(
        name="bleu_cy",
        sources=['bleu_cy.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=['./cpp_sources/'],
    )
]

setup(
    name='cy_bleu_setup',
    ext_modules=cythonize(ext_modules, build_dir='build'),
    requires=['Cython']
)
