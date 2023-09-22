#!/bin/bash
pip3 install ./packages/TopsInference-2.4.12-py3.8-none-any.whl

dpkg -i ./sdk_installers/topsruntime_2.4.12-1_amd64.deb
dpkg -i ./sdk_installers/tops-sdk_2.4.12-1_amd64.deb

