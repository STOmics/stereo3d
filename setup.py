# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requires = f.read().splitlines()

# Parse version number from cellbin2/__init__.py:
with open('stereo3d/__init__.py') as f:
    info = {}
    for line in f:
        if line.startswith('__version__'):
            exec(line, info)
            break

print(f"Version: {info['__version__']}")


setup(
    name='stereo-3d',
    version=info['__version__'],
    description='.',
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    packages=find_packages(),
    author='stereo-3d research group',
    author_email='bgi@genomics.cn',
    url='https://github.com/STOmics/stereo3d',
    install_requires=requires,
    python_requires='==3.8.*',
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
    ],

  )
