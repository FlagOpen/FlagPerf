#!/bin/bash

if [ $# -lt 1 ]; then
	echo "Usage: ./gen_docker_image.sh <image_tag>"
	echo "Please provide docker image tag"
	exit 1;
fi

IMAGE_NAME=$1
IMAGE_NAME_TEMP=${IMAGE_NAME}_temp
CONTAINER_NAME_TEMP=${IMAGE_NAME/:/_}_temp

docker build -t ${IMAGE_NAME_TEMP} ./

source ./image.conf

BASE_DIR=$( cd -P "$(dirname "$0")" && pwd)
PACKAGE_DIR=$( cd -P "${HOST_PACKAGE_DIR}" && pwd)
SDK_INSTALLER_DIR=$( cd -P "${HOST_SDK_INSTALLER_DIR}" && pwd)

echo "CURRENT_DIR = $BASE_DIR"
echo "PACKAGE_DIR = $PACKAGE_DIR"
echo "SDK_INSTALLER_DIR = ${SDK_INSTALLER_DIR}"

docker container rm -f "${CONTAINER_NAME_TEMP}" >/dev/null || true

docker run --rm --init --detach --net=host --uts=host --ipc=host \
	--security-opt=seccomp=unconfined --privileged=true \
	--ulimit=stack=67108864 --ulimit=memlock=-1 \
	-v ${BASE_DIR}:/installer/bin \
	-v ${PACKAGE_DIR}:/installer/packages \
	-v ${SDK_INSTALLER_DIR}:/installer/sdk_installers \
	-v /dev:/dev \
	-v /usr/src/:/usr/src \
	-v /lib/modules/:/lib/modules \
	--cap-add=ALL \
	--name=${CONTAINER_NAME_TEMP} ${IMAGE_NAME_TEMP} sleep infinity

docker exec -it "${CONTAINER_NAME_TEMP}" /bin/bash -c "bash /installer/bin/install.sh"

docker commit -a "baai" -m "mlperf test"  ${CONTAINER_NAME_TEMP} ${IMAGE_NAME}

docker container rm -f "${CONTAINER_NAME_TEMP}"
docker rmi -f "${IMAGE_NAME_TEMP}"
