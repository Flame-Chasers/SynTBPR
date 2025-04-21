import torch
_ = torch.manual_seed(123)
from torchmetrics.image.inception import InceptionScore
inception = InceptionScore()
# generate some images
imgs = torch.randint(0, 255, (100, 3, 299, 299), dtype=torch.uint8)
inception.update(imgs)
inception_score = inception.compute()
print(inception_score)