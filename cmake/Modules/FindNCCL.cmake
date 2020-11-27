set(NCCL_INC_PATHS
    /usr/include
    /usr/local/include
    $ENV{NCCL_DIR}/include
    )

set(NCCL_LIB_PATHS
    /lib
    /lib64
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    $ENV{NCCL_DIR}/lib
    )

find_path(NCCL_INCLUDE_DIR NAMES nccl.h PA