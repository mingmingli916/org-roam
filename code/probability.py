import torch
from torch.distributions import multinomial

fair_probs = torch.ones([6]) / 6
fair_probs
# draw a single sample
multinomial.Multinomial(1, fair_probs).sample()
# draw 3 samples
multinomial.Multinomial(3, fair_probs).sample()
