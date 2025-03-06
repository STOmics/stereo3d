from distutils.core import setup
from Cython.Build import cythonize
from setuptools import setup, find_packages

setup(
    ext_modules = cythonize("xxx.py")
)

setup(
    name="stereo3d",
    version="v1.0.0",
    author="stereomics.genomics.cn",
    author_email="",
    description="Auto get stereo 3D data with Python",
    url="", 
    packages=find_packages(),
    package_data={
        '':['*.py'],
        'bandwidth_reporter':['*.txt']
    }
)