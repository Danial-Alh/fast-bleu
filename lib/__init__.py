def compile_cython():
    import os, inspect
    ROOT_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../'
    curdir = os.curdir
    os.chdir(ROOT_PATH)
    return_value = os.system("python3.6 setup.py build_ext --build-lib ./lib".format())
    if return_value != 0:
        raise BaseException('compilation failed!')
    os.chdir(curdir)


compile_cython()
