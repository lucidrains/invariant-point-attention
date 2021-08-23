import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from contextlib import contextmanager
from torch import nn, einsum

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

@contextmanager
def disable_tf32():
    orig_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = orig_value

# classes

class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        scalar_key_dim = 16,
        scalar_value_dim = 16,
        point_key_dim = 4,
        point_value_dim = 4,
        pairwise_repr_dim = None,
        require_pairwise_repr = True,
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.require_pairwise_repr = require_pairwise_repr

        # num attention contributions

        num_attn_logits = 3 if require_pairwise_repr else 2

        # qkv projection for scalar attention (normal)

        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5

        self.to_scalar_q = nn.Linear(dim, scalar_key_dim * heads, bias = False)
        self.to_scalar_k = nn.Linear(dim, scalar_key_dim * heads, bias = False)
        self.to_scalar_v = nn.Linear(dim, scalar_value_dim * heads, bias = False)

        # qkv projection for point attention (coordinate and orientation aware)

        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.)) - 1.)
        self.point_weights = nn.Parameter(point_weight_init_value)

        self.point_attn_logits_scale = ((num_attn_logits * point_key_dim) * (9 / 2)) ** -0.5

        self.to_point_q = nn.Linear(dim, point_key_dim * heads * 3, bias = False)
        self.to_point_k = nn.Linear(dim, point_key_dim * heads * 3, bias = False)
        self.to_point_v = nn.Linear(dim, point_value_dim * heads * 3, bias = False)

        # pairwise representation projection to attention bias

        pairwise_repr_dim = default(pairwise_repr_dim, dim) if require_pairwise_repr else 0

        if require_pairwise_repr:
            self.pairwise_attn_logits_scale = num_attn_logits ** -0.5

            self.to_pairwise_attn_bias = nn.Sequential(
                nn.Linear(pairwise_repr_dim, heads),
                Rearrange('b ... h -> (b h) ...')
            )

        # combine out - scalar dim + pairwise dim + point dim * (3 for coordinates in R3 and then 1 for norm)

        self.to_out = nn.Linear(heads * (scalar_value_dim + pairwise_repr_dim + point_value_dim * (3 + 1)), dim)

    def forward(
        self,
        single_repr,
        pairwise_repr = None,
        *,
        rotations,
        translations,
        mask = None
    ):
        x, b, h, eps, require_pairwise_repr = single_repr, single_repr.shape[0], self.heads, self.eps, self.require_pairwise_repr
        assert not (require_pairwise_repr and not exists(pairwise_repr)), 'pairwise representation must be given as second argument'

        # get queries, keys, values for scalar and point (coordinate-aware) attention pathways

        q_scalar, k_scalar, v_scalar = self.to_scalar_q(x), self.to_scalar_k(x), self.to_scalar_v(x)

        q_point, k_point, v_point = self.to_point_q(x), self.to_point_k(x), self.to_point_v(x)

        # split out heads

        q_scalar, k_scalar, v_scalar = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q_scalar, k_scalar, v_scalar))
        q_point, k_point, v_point = map(lambda t: rearrange(t, 'b n (h d c) -> (b h) n d c', h = h, c = 3), (q_point, k_point, v_point))

        rotations = repeat(rotations, 'b n r1 r2 -> (b h) n r1 r2', h = h)
        translations = repeat(translations, 'b n c -> (b h) n () c', h = h)

        # rotate qkv points into global frame

        q_point = einsum('b n d c, b n c r -> b n d r', q_point, rotations) + translations
        k_point = einsum('b n d c, b n c r -> b n d r', k_point, rotations) + translations
        v_point = einsum('b n d c, b n c r -> b n d r', v_point, rotations) + translations

        # derive attn logits for scalar and pairwise

        attn_logits_scalar = einsum('b i d, b j d -> b i j', q_scalar, k_scalar) * self.scalar_attn_logits_scale

        if require_pairwise_repr:
            attn_logits_pairwise = self.to_pairwise_attn_bias(pairwise_repr) * self.pairwise_attn_logits_scale

        # derive attn logits for point attention

        point_qk_diff = rearrange(q_point, 'b i d c -> b i () d c') - rearrange(k_point, 'b j d c -> b () j d c')
        point_dist = (point_qk_diff ** 2).sum(dim = -2)

        point_weights = F.softplus(self.point_weights)
        point_weights = repeat(point_weights, 'h -> (b h) () () ()', b = b)

        attn_logits_points = -0.5 * (point_dist * point_weights * self.point_attn_logits_scale).sum(dim = -1)

        # combine attn logits

        attn_logits = attn_logits_scalar + attn_logits_points

        if require_pairwise_repr:
            attn_logits = attn_logits + attn_logits_pairwise

        # mask

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h = h)
            mask_value = max_neg_value(attn_logits)
            attn_logits = attn_logits.masked_fill(~mask, mask_value)

        # attention

        attn = attn_logits.softmax(dim = - 1)

        with disable_tf32(), autocast(enabled = False):
            # disable TF32 for precision

            # aggregate values

            results_scalar = einsum('b i j, b j d -> b i d', attn, v_scalar)

            attn_with_heads = rearrange(attn, '(b h) i j -> b h i j', h = h)

            if require_pairwise_repr:
                results_pairwise = einsum('b h i j, b i j d -> b h i d', attn_with_heads, pairwise_repr)

            # aggregate point values

            results_points = einsum('b i j, b j d c -> b i d c', attn, v_point)

            # rotate aggregated point values back into local frame

            results_points = einsum('b n d c, b n c r -> b n d r', results_points - translations, rotations.transpose(-1, -2))
            results_points_norm = torch.sqrt( torch.square(results_points).sum(dim=-1) + eps )

        # merge back heads

        results_scalar = rearrange(results_scalar, '(b h) n d -> b n (h d)', h = h)
        results_points = rearrange(results_points, '(b h) n d c -> b n (h d c)', h = h)
        results_points_norm = rearrange(results_points_norm, '(b h) n d -> b n (h d)', h = h)

        results = (results_scalar, results_points, results_points_norm)

        if require_pairwise_repr:
            results_pairwise = rearrange(results_pairwise, 'b h n d -> b n (h d)', h = h)
            results = (*results, results_pairwise)

        # concat results and project out

        results = torch.cat(results, dim = -1)
        return self.to_out(results)

# one transformer block based on IPA

def FeedForward(dim, mult = 1., num_layers = 2, act = nn.ReLU):
    layers = []
    dim_hidden = dim * mult

    for ind in range(num_layers):
        is_first = ind == 0
        is_last  = ind == (num_layers - 1)
        dim_in   = dim if is_first else dim_hidden
        dim_out  = dim if is_last else dim_hidden

        layers.append(nn.Linear(dim_in, dim_out))

        if is_last:
            continue

        layers.append(act())

    return nn.Sequential(*layers)

class IPABlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        ff_mult = 1,
        ff_num_layers = 3,     # in the paper, they used 3 layer transition (feedforward) block
        post_norm = True,      # in the paper, they used post-layernorm - offering pre-norm as well
        **kwargs
    ):
        super().__init__()
        self.post_norm = post_norm

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = InvariantPointAttention(dim = dim, **kwargs)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult = ff_mult, num_layers = ff_num_layers)

    def forward(self, x, **kwargs):
        post_norm = self.post_norm

        attn_input = x if post_norm else self.attn_norm(x)
        x = self.attn(attn_input, **kwargs) + x
        x = self.attn_norm(x) if post_norm else x

        ff_input = x if post_norm else self.ff_norm(x)
        x = self.ff(ff_input) + x
        x = self.ff_norm(x) if post_norm else x
        return x

# add an IPA Transformer - iteratively updating rotations and translations

# this portion is not accurate to AF2, as AF2 applies a FAPE auxiliary loss on each layer, as well as a stop gradient on the rotations
# just an attempt to see if this could evolve to something more generally usable

class IPATransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_tokens = None,
        predict_points = False,
        **kwargs
    ):
        super().__init__()

        # using quaternion functions from pytorch3d

        try:
            from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix
            self.quaternion_to_matrix = quaternion_to_matrix
            self.quaternion_multiply = quaternion_multiply
        except ImportError as err:
            print('unable to import pytorch3d - please install with `conda install pytorch3d -c pytorch3d`')
            raise err

        # embedding

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None

        # layers

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                IPABlock(dim = dim, **kwargs),
                nn.Linear(dim, 6)
            ]))

        # output

        self.predict_points = predict_points

        if predict_points:
            self.to_points = nn.Linear(dim, 3)

    def forward(
        self,
        single_repr,
        *,
        translations = None,
        quaternions = None,
        pairwise_repr = None,
        mask = None
    ):
        x, device, quaternion_multiply, quaternion_to_matrix = single_repr, single_repr.device, self.quaternion_multiply, self.quaternion_to_matrix
        b, n, *_ = x.shape

        if exists(self.token_emb):
            x = self.token_emb(x)

        # if no initial quaternions passed in, start from identity

        if not exists(quaternions):
            quaternions = torch.tensor([1., 0., 0., 0.], device = device) # initial rotations
            quaternions = repeat(quaternions, 'd -> b n d', b = b, n = n)

        # if not translations passed in, start from identity

        if not exists(translations):
            translations = torch.zeros((b, n, 3), device = device)

        # go through the layers and apply invariant point attention and feedforward

        for block, to_update in self.layers:
            rotations = quaternion_to_matrix(quaternions)

            x = block(
                x,
                pairwise_repr = pairwise_repr,
                rotations = rotations,
                translations = translations
            )

            # update quaternion and translation

            quaternion_update, translation_update = to_update(x).chunk(2, dim = -1)
            quaternion_update = F.pad(quaternion_update, (1, 0), value = 1.)

            quaternions = quaternion_multiply(quaternions, quaternion_update)
            translations = translations + einsum('b n c, b n c r -> b n r', translation_update, rotations)

        if not self.predict_points:
            return x, translations, quaternions

        points_local = self.to_points(x)
        rotations = quaternion_to_matrix(quaternions)
        points_global = einsum('b n c, b n c d -> b n d', points_local, rotations) + translations
        return points_global
