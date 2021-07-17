<img src="./ipa.png" width="400px"></img>

## Invariant Point Attention - Pytorch

Implementation of Invariant Point Attention as a standalone module, which was used in the structure module of <a href="https://github.com/deepmind/alphafold">Alphafold2</a> for coordinate refinement.

- [ ] write up a test for invariance under rotation

## Install

```bash
$ pip install invariant-point-attention
```

## Usage

```python
import torch
from einops import repeat
from invariant_point_attention import InvariantPointAttention

attn = InvariantPointAttention(
    dim = 64,                  # single (and pairwise) representation dimension
    heads = 8,                 # number of attention heads
    scalar_key_dim = 16,       # scalar query-key dimension
    scalar_value_dim = 16,     # scalar value dimension
    point_key_dim = 4,         # point query-key dimension
    point_value_dim = 4        # point value dimension
)

single_repr   = torch.randn(1, 256, 64)      # (batch x seq x dim)
pairwise_repr = torch.randn(1, 256, 256, 64) # (batch x seq x seq x dim)
mask          = torch.ones(1, 256).bool()    # (batch x seq)

rotations     = repeat(torch.eye(3), 'r s -> b n r s', b = 1, n = 256)  # (batch x seq x rot1 x rot2) - example is identity
translations  = torch.zeros(1, 256, 3) # translation, also identity for example

attn_out = attn(
    single_repr,
    pairwise_repr,
    rotations = rotations,
    translations = translations,
    mask = mask
)

attn_out.shape # (1, 256, 64)
```
## Citations

```bibtex
@Article{AlphaFold2021,
    author  = {Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\v{Z}}{\'\i}dek, Augustin and Potapenko, Anna and Bridgland, Alex and Meyer, Clemens and Kohl, Simon A A and Ballard, Andrew J and Cowie, Andrew and Romera-Paredes, Bernardino and Nikolov, Stanislav and Jain, Rishub and Adler, Jonas and Back, Trevor and Petersen, Stig and Reiman, David and Clancy, Ellen and Zielinski, Michal and Steinegger, Martin and Pacholska, Michalina and Berghammer, Tamas and Bodenstein, Sebastian and Silver, David and Vinyals, Oriol and Senior, Andrew W and Kavukcuoglu, Koray and Kohli, Pushmeet and Hassabis, Demis},
    journal = {Nature},
    title   = {Highly accurate protein structure prediction with {AlphaFold}},
    year    = {2021},
    doi     = {10.1038/s41586-021-03819-2},
    note    = {(Accelerated article preview)},
}
```
