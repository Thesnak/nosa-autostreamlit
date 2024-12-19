from setuptools import setup, find_packages

setup(
    name='nosa-autostreamlit',
    version='0.1.0',
    author='Mohamed Mahmoud',
    author_email='mohamed.mahmoud0726@gmail.com',
    description='An automated Streamlit application development framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thesnak/nosa-autostreamlit',
    packages=find_packages(),
    install_requires=[
      "streamlit>=1.41.1",
        "pandas",
        "numpy",
        "plotly",
        "scikit-learn>=1.6.0",
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'nosa-autostreamlit=nosa_autostreamlit.cli:main',
        ],
    },
    extras_require={
        'dev': [
            'pytest',
            'streamlit',
            'pandas',
            'numpy'
        ]
    },

)