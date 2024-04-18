from setuptools import find_packages, setup


setup(
    name='glove',
    packages=find_packages(include=['glove']),
    version='0.1.0',
    description='Generalized LOtka Volterra Estimator',
    author='Jaron Thompson',
    license='MIT',
    install_requires=['numpy',
                      'pandas',
                      'jax',
                      'jaxlib']
)
