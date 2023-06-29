from setuptools import setup, find_packages

setup(
    name='quacc_tests',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'quacc_tests=quacc_tests.main:main',
        ],
    },
)

