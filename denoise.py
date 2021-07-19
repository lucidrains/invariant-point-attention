import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.optim import Adam

from einops import rearrange, repeat
import sidechainnet as scn
from invariant_point_attention import IPATransformer

BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 16

def cycle(loader, len_thres = 200):
    while True:
        for data in loader:
            if data.seqs.shape[1] > len_thres:
                continue
            yield data

net = IPATransformer(
    dim = 16,
    num_tokens = 21,
    depth = 5,
    require_pairwise_repr = False,
    predict_points = True
).cuda()

data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = BATCH_SIZE,
    dynamic_batching = False
)

dl = cycle(data['train'])
optim = Adam(net.parameters(), lr=1e-3)

for _ in range(10000):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(dl)
        seqs, coords, masks = batch.seqs, batch.crds, batch.msks

        seqs = seqs.cuda().argmax(dim = -1)
        coords = coords.cuda()
        masks = masks.cuda().bool()

        l = seqs.shape[1]
        coords = rearrange(coords, 'b (l s) c -> b l s c', s = 14)

        # Keeping only the backbone coordinates

        coords = coords[:, :, 0:3, :]
        coords = rearrange(coords, 'b l s c -> b (l s) c')

        seq = repeat(seqs, 'b n -> b (n c)', c = 3)
        masks = repeat(masks, 'b n -> b (n c)', c = 3)

        noised_coords = coords + torch.randn_like(coords)

        denoised_coords = net(
            seq,
            translations = noised_coords,
            mask = masks
        )

        loss = F.mse_loss(denoised_coords[masks], coords[masks])
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print('loss:', loss.item())
    optim.step()
    optim.zero_grad()
