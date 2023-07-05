from setuptools import setup, find_packages

setup(
    name='bo_icp',
    version='0.1',
    url='http://github.com/arpg/bo-icp.git',
    packages=find_packages(),
    author='Harel Biggie',
    author_email="harel.biggie@colorado.com",
    description='BO-ICP',
    long_description='Initialization of Iterative Closest Point Based on Bayesian Optimization',
    download_url='https://github.com/arpg/bo-icp',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.18.0",
        "open3d >= 0.14",
    ],
    classifiers=[
        'License :: OSI Approved :: Apache2 License',
    ]
)
