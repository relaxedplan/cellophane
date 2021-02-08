from setuptools import setup, find_packages

setup(
    name='cellophane',
    version='0.1.0',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description="Auditing tool for use when protected class membership is unobserved in data",
    long_description=open('README.md').read(),
    install_requires=["setuptools >= 40.6.0", "wheel","matplotlib>=3.1.3","numpy>=1.18.1","pandas>=1.0.1","seaborn>=0.10.0","scikit-learn>=0.23.1"],
    url='https://github.com/relaxedplan/cellophane',
    author='Michael McKenna',
    author_email='michael.mckenna95@gmail.com'
)