#! /bin/bash
# pip install -e .
python setup.py develop
pip install motmetrics==1.1.3
pip install neptune-client
export PYTHONPATH="${PYTHONPATH}:/btherien/github/nuscenes-devkit/python-sdk"