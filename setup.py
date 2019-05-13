import setuptools

setuptools.setup(
    name='causaldag',
    version='0.1a.98',
    description='Causal DAG manipulation and inference',
    long_description='CausalDAG is a Python package for the creation, manipulation, and learning of Causal DAGs.',
    author='Chandler Squires',
    author_email='chandlersquires18@gmail.com',
    packages=setuptools.find_packages(exclude=['tests']),
    python_requires='>3.5.0',
    zip_safe=False,
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[
        'scipy',
        'dataclasses',
        'numpy',
        # 'scikit_sparse',
        'matplotlib',
        'pandas',
        'scikit_learn',
        'typing',
        'pygam'
    ]
)

