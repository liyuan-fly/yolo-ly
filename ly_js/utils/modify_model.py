
# 自定义模型输出头
def modify_head(model=None):
    backbone = model.model[:-1]
    head = model.model[-1]

    return model