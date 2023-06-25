
get_modelzoo_base_code_by_git(){
    [ -z $git_url  ] && { echo "args git_url not exist";return 1; }
    [ -z $branch  ] && { echo "args branch not exist";return 1; }
    [ -z $modelzoo_sub_dir  ] && { echo "args modelzoo_sub_dir not exist";return 1; }
    [ -z $commitid  ] && { echo "args commitid not exist";return 1; }

    git clone $git_url -b $branch || { echo "warn git clone failed"; return 1; }
    code_dir=${modelzoo_sub_dir%%/*}

    cd ${code_dir}
    git reset --hard $commitid || { echo "warn git reset failed"; return 1; }
    cd -
}

make_patch(){
    [ -z $BUILD_TMP_PATH  ] && { echo "args BUILD_TMP_PATH not exist";return 1; }
    [ -z $modelzoo_sub_dir  ] && { echo "args modelzoo_sub_dir not exist";return 1; }
    [ -z $target_dir  ] && { echo "args target_dir not exist";return 1; }
    [ -z $patch_file_name  ] && { echo "args patch_file_name not exist";return 1; }
    [ -z $CUR_PATH  ] && { echo "args patch_file_name not exist";return 1; }

    cd $BUILD_TMP_PATH
    get_modelzoo_base_code_by_git  || { echo "warn git getcode failed"; return 1; }
    cp $modelzoo_sub_dir -rf   $BUILD_TMP_PATH/origin
    cp $target_dir -rf $BUILD_TMP_PATH/code

    diff -Nur origin code > $BUILD_TMP_PATH/$patch_file_name.patch

    cp $BUILD_TMP_PATH/$patch_file_name.patch $CUR_PATH/
}

load_code(){
    [ -z $BUILD_TMP_PATH  ] && { echo "args BUILD_TMP_PATH not exist";return 1; }
    [ -z $modelzoo_sub_dir  ] && { echo "args modelzoo_sub_dir not exist";return 1; }
    [ -z $patch_file_name  ] && { echo "args patch_file_name not exist";return 1; }
    [ -z $target_patchcode_dir  ] && { echo "args target_patchcode_dir not exist";return 1; }
    [ -z $CUR_PATH  ] && { echo "args CUR_PATH not exist";return 1; }

    cd $BUILD_TMP_PATH
    get_modelzoo_base_code_by_git || { echo "warn git getcode failed"; return 1; }
    cp $modelzoo_sub_dir -rf   $BUILD_TMP_PATH/origin
    cp $modelzoo_sub_dir -rf   $BUILD_TMP_PATH/code

    if [ -f $CUR_PATH/$patch_file_name.patch ];then
        patch -p0 < $CUR_PATH/$patch_file_name.patch || { echo "warn patch pfile failed"; return 1; }
    else
        echo "no patch file"
    fi
    [ ! -d $target_patchcode_dir ] || rm -rf $target_patchcode_dir
    mkdir $target_patchcode_dir
    cp $BUILD_TMP_PATH/code/* -rf  $target_patchcode_dir/
}

mk_version_file()
{
    version_files=$1
    echo "git_url: $git_url" > $version_files
    echo "branch: $branch" >> $version_files
    echo "commitid: $commitid" >> $version_files
    if [ -f $CUR_PATH/$patch_file_name.patch ];then
        echo "patch_file_name: $patch_file_name.patch" >> $version_files
    else
        echo "patch_file_name: None" >> $version_files
    fi

}
