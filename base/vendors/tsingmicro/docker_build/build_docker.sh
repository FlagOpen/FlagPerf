if [ -e "../TX81" ]; then
	tar czf TX81.tar.gz ../TX81
else
	mv Tsingmicro_*.Toolkit.tar.gz TX81.tar.gz
fi
docker build -t tx8-ubuntu:rex1032_flagperf .
