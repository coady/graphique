from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

ext_module = Extension(
    'graphique.arrayed',
    sources=['graphique/arrayed' + ('.pyx' if cythonize else '.cpp')],
    extra_compile_args=['-std=c++11'],
    extra_link_args=['-std=c++11'],
)

setup(
    name='graphique',
    version='0.1',
    description='GraphQL service for arrow tables and parquet files.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Aric Coady',
    author_email='aric.coady@gmail.com',
    url='https://github.com/coady/graphique',
    project_urls={'Documentation': 'https://graphique.readthedocs.io'},
    license='Apache Software License',
    packages=['graphique'],
    package_data={'graphique': ['py.typed']},
    zip_safe=False,
    ext_modules=cythonize([ext_module]) if cythonize else [ext_module],
    install_requires=['pyarrow>=1', 'strawberry-graphql>=0.30'],
    extras_require={'docs': ['recommonmark', 'sphinx'], 'timestamp': ['pytz']},
    python_requires='>=3.7',
    tests_require=['pytest-cov', 'requests'],
    keywords='graphql arrow parquet',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Typing :: Typed',
    ],
)
