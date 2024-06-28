source /usr/local/Ascend/toolbox/set_env.sh
for i in {0..7}
do 
    ascend-dmi --bw -t d2h -d $i -s 536870912 --et 10
done
