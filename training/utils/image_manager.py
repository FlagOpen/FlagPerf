# Copyright  2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
''' Local sudo docker image manger.
usage:
image_management.py -o [operation] -i [repository] -t [tag]
'''
import os
import sys
import argparse
from run_cmd import run_cmd_wait as rcw
from container_manager import ContainerManager


def _parse_args():
    ''' Check script input parameter. '''
    help_message = '''Operations for docker image:
exist     Whether the image exists
remove    Remove a docker image
build     Build a docker image with two options if the image doesn't exist:
          -d [directory]  Directory contains dockerfile and install script
          -f [framework]  AI framework '''

    parser = argparse.ArgumentParser(
        description='Docker managment script',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-o',
                        type=str,
                        metavar='[operation]',
                        required=True,
                        choices=['exist', 'remove', 'build'],
                        help=help_message)
    parser.add_argument('-i',
                        type=str,
                        metavar='[repository]',
                        required=True,
                        help='image repository')
    parser.add_argument('-t',
                        type=str,
                        metavar='[tag]',
                        required=True,
                        help='image tag')
    args, _ = parser.parse_known_args()
    if args.o == "build":
        parser.add_argument("-d",
                            type=str,
                            required=True,
                            help="dir contains dockerfile for building image.")
        parser.add_argument("-f",
                            type=str,
                            required=True,
                            help="testcase framework of the image.")
    args = parser.parse_args()
    return args


class ImageManager():
    '''Local image manager.
    Support operations below:
        -- remove,          rm image from local
        -- exists,     query if image exist local
        -- build_image,     build docker image
    '''

    def __init__(self, repository, tag):
        self.repository = repository
        self.tag = tag

    def exist(self):
        '''Check if local image existi or not
        Return code:
            0 - image already exist
            1 - image doesn't exist
        '''
        cmd = "sudo docker images|grep -w \"" + self.repository + "\"|grep -w \"" + \
              self.tag + "\""
        print(cmd)
        print(rcw(cmd, 10, retouts=False))
        if rcw(cmd, 10, retouts=False) != 0:
            return 1
        return 0

    def remove(self):
        '''Remove local image
           Return code:
            0  - rm image successfully
            1  - rm image failed
        '''
        cmd = "sudo docker rmi " + self.repository + ":" + self.tag
        if rcw(cmd, 60, retouts=False) != 0:
            return 1
        return 0

    def _rm_tmp_image(self, tmp_image_name, cont_mgr):
        '''remove temp container and temp image.'''
        clean_tmp_cmd = "docker rmi -f " + tmp_image_name
        cont_mgr.remove()
        rcw(clean_tmp_cmd, 30, retouts=False)

    def build_image(self, image_dir, framework):
        '''Build docker image in vendor's path.
        '''
        # First, build base docker image.
        tmp_image_name = "tmp_" + self.repository + ":" + self.tag
        build_cmd = "cd " + image_dir + " && docker build -t " \
                    + tmp_image_name + " ./"
        if rcw(build_cmd, 600, retouts=False) != 0:
            print("docker build failed. " + tmp_image_name)
            return 1

        # Second, start a container with the base image
        tmp_container_name = "tmp_" + self.repository + "-" + self.tag \
                             + "-container"
        image_dir_in_container = "/workspace/docker_image"
        start_args = " --rm --init --detach --net=host --uts=host " \
                     + "--ipc=host --security-opt=seccomp=unconfined " \
                     + "--privileged=true --ulimit=stack=67108864 " \
                     + "--ulimit=memlock=-1 -v " + image_dir + ":" \
                     + image_dir_in_container
        cont_mgr = ContainerManager(tmp_container_name)
        cont_mgr.remove()
        ret, outs = cont_mgr.run_new(start_args, tmp_image_name)
        if ret != 0:
            print("Start new container with base image failed.")
            print("Error: " + outs[0])
            self._rm_tmp_image(tmp_image_name, cont_mgr)
            return ret

        # Third, install packages in container.
        install_script = framework + "_install.sh"
        if not os.path.isfile(os.path.join(image_dir, install_script)):
            print("Can't find <framework>_install.sh")
            install_cmd = ":"
        else:
            install_cmd = "bash " + image_dir_in_container + "/" \
                          + install_script
        ret, outs = cont_mgr.run_cmd_in(install_cmd, 1800, detach=False)
        if ret != 0:
            print("Run install command in temp container failed.")
            print("Error: " + outs[0])
            self._rm_tmp_image(tmp_image_name, cont_mgr)
            return ret
        commit_cmd = "docker commit -a \"baai\" -m \"flagperf training\" " \
                     + tmp_container_name + " " + self.repository + ":" \
                     + self.tag

        ret, outs = rcw(commit_cmd, 30)
        if ret != 0:
            print("Commit docker image failed.")
            print("Error: " + outs[0])
            self._rm_tmp_image(tmp_image_name, cont_mgr)
            return ret

        # At last, remove temp container and temp image.
        self._rm_tmp_image(tmp_image_name, cont_mgr)
        return 0


def main():
    '''Main process to manage image
    Return code:
        0 - successfull.
        1 - failed.
        2 - invalid operation. '''
    args = _parse_args()
    operation = args.o
    image = args.i
    tag = args.t

    image_manager = ImageManager(image, tag)
    if operation == "exist":
        ret = image_manager.exist()
        if ret == 0:
            print("Doker image exists.")
        else:
            print("Doker image doesn't exist.")
    elif operation == "remove":
        ret = image_manager.remove()
        if ret == 0:
            print("Remove doker image successfully.")
        else:
            print("Remove doker image failed.")
    elif operation == "build":
        if image_manager.exist() == 0:
            ret = 0
        else:
            image_dir = args.d
            framework = args.f
            ret = image_manager.build_image(image_dir, framework)
    else:
        print("Invalid operation.")
        sys.exit(2)
    sys.exit(ret)


if __name__ == "__main__":
    main()
