import os.path as ospath


MODEL_DIR = ospath.abspath(
    ospath.join(
        __file__,
        "../../../"
    )
)

CURRENT_MODEL_NAME = ospath.basename(MODEL_DIR)

PROJ_DIR = ospath.abspath(
    ospath.join(
        MODEL_DIR,
        "../../"
    )
)
