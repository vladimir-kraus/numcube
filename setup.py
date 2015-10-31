from numcube import __version__
from distutils.core import setup

setup(
    name='numcube',
    version=__version__,
    author='Vladimir Kraus',
    author_email='vladimir.kraus@gmail.com',
    packages=['numcube'],
    url='http://github.com/vladimir-kraus/numcube',
    license='MIT',
    description='...',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy >= 1.8.0'
    ],
)
