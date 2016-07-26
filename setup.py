#!/usr/bin/env python

import setuptools

install_requires = ["numpy", "pystan"]

setuptools.setup(
    name = 'stark',
    version = '0.0.1',
    license = 'Apache',
    description = 'Distributed Inference with Stan & Spark.',
    author = 'Homer Strong',
    author_email = 'homer.strong@gmail.com',
    url = 'https://github.com/strongh/stark',
    platforms = 'any',
    packages = ['stark'],
    zip_safe = True,
    verbose = False,
    install_requires = install_requires
)
