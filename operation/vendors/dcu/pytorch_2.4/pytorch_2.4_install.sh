mkdir -p $FLAGGEMS_WORK_DIR && cd $FLAGGEMS_WORK_DIR
rm -rf FlagGems
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems 
git checkout v2.0-perf-hip
pip install pytest
pip install scipy
pip install -e . --no-deps
/etc/init.d/ssh restart