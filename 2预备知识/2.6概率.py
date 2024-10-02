from torch.distributions import multinomial
import torch
from matplotlib import pyplot as plt
import numpy as np

probs = torch.ones(6)
# total_count 为抽样次数，probs为样本，是一个tensor
multinomial_distribution = multinomial.Multinomial(total_count=1, probs=probs)

# 采样
print(multinomial_distribution.sample())
# 对数概率分布
print(multinomial_distribution.logits)