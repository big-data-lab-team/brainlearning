import sys
from setuptools import setup
import sys

VERSION = "0.0.1"
DEPS = ["Keras", "nibabel", "numpy", ]

setup(name="brainlearning",
      version=VERSION,
      description=" Machine-learning experiments with neuroimaging pipelines",
      url="https://github.com/big-data-lab-team/brainlearning",
      author="Nikita Baranov",
      classifiers=[
                "Programming Language :: Python",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.4",
                "Programming Language :: Python :: 3.5",
                "Programming Language :: Python :: 3.6",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: Implementation :: PyPy",
                "License :: OSI Approved :: MIT License",
                "Topic :: Software Development :: Libraries :: Python Modules",
                "Operating System :: OS Independent",
                "Natural Language :: English"
                  ],
      license="GPLv3",
      packages=["brainlearning"],
      include_package_data=True,
      test_suite="pytest",
      tests_require=["pytest"],
      setup_requires=DEPS,
      install_requires=DEPS,
      zip_safe=False)
