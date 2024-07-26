# When using Conda, please specify the location of your environment variable
# If you are using the system's environment, there is no need to set this variable
source /root/anaconda3/bin/activate /opt/nvme1n1/conda-envs/patch; 

# Activate the environment related to Ascend
source /usr/local/Ascend/driver/bin/setenv.bash; 
source /usr/local/Ascend/ascend-toolkit/set_env.sh
