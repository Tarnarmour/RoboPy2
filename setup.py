from setuptools import setup, find_packages


setup(
    name='robopy2',
    version='0.1.0',
    description='Lightweight pure python robotics package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='John Morrell',
    author_email='Tarnarmour@gmail.com',
    url='https://github.com/Tarnarmour/RoboPy2',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.25.0'
    ],
    extras_require={
        'dev': ['pytest']
    },
    python_requires='>=3.11'
)
