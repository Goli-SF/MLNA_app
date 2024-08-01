from setuptools import setup, find_packages

with open('requirements.txt') as f:
    content= f.readlines()
requirements= [x.strip() for x in content]

setup (name= 'mlna',
       description= 'the multilingual network analysis package',
       install_requires= requirements,
       packages= find_packages()
       )
