from distutils.core import setup

setup(
    name='numcube',
    version='0.1.0',
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
