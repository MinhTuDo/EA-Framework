from setuptools import setup, find_packages

setup(
    name='ea',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 'keras', 'tensorflow', 'matplotlib', 'pandas'
    ],
    author='tu do',
    author_email='tu.dominh2k@gmail.com',
    include_package_data=True,
    zip_safe=False
)