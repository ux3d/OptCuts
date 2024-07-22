import os
import sys
import subprocess
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            "-DCMAKE_BUILD_TYPE=Release"
        ]
        build_args = []

        if not os.path.exists(ext.build_temp):
            os.makedirs(ext.build_temp)
        subprocess.check_call(['cmake', '..'] + cmake_args, cwd=ext.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'install'] + build_args, cwd=ext.build_temp)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='.', builddir='build'):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.build_temp = os.path.abspath(builddir)
        print(self.build_temp)

setup(
    name='PyOptCuts',
    version='0.1.0',
    author='Moritz Becher',
    author_email='becher@ux3d.io',
    description='A Python wrapper for OptCuts',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=[CMakeExtension('PyOptCuts._py_opt_cuts', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    packages=setuptools.find_packages(),
    include_package_data=True,
)
