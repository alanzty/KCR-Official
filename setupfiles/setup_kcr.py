import os
import os.path as osp
import shutil
import sys
import warnings
from setuptools import find_packages, setup



if __name__ == '__main__':
    setup(
        name='kcr',
        version='0.0.1',
        description='Knowledge Combination for Rotated Object Detection',
        long_description='Knowledge Combination for Rotated Object Detection',
        long_description_content_type='text/markdown',
        author='Alan',
        author_email='tianyuzhu52@gmail.com',
        keywords='computer vision, object detection, rotation detection',
        url='https://github.com/open-mmlab/mmrotate',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        include_package_data=True,

        zip_safe=False)