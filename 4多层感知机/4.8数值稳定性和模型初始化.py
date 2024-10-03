from torch import nn
# 初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)  # 均匀分布的 Xavier 初始化
        # nn.init.xavier_normal_(m.weight)  # 正态分布的 Xavier 初始化
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # 偏置项初始化为 0