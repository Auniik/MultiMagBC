

import torch

from backbones.our.model import MMNet


dummy_batch = {
    f'mag_{mag}': torch.randn(2, 3, 224, 224) for mag in ['40', '100', '200', '400']
}
model = MMNet().to('cpu')
output = model(dummy_batch)
print(output.shape)