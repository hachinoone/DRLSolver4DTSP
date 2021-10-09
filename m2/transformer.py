import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math
from typing import NamedTuple
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _mask_long2byte(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n].to(torch.bool).view(*mask.size()[:-1], -1)[..., :n]

def _mask_byte2bool(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[..., :n] > 0
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[..., :n] > 0

def mask_long2bool(mask, n=None):
    assert mask.dtype == torch.int64
    return _mask_byte2bool(_mask_long2byte(mask), n=n)


def mask_long_scatter(mask, values, check_unset=True):
    """
    Sets values in mask in dimension -1 with arbitrary batch dimensions
    If values contains -1, nothing is set
    Note: does not work for setting multiple values at once (like normal scatter)
    """
    assert mask.size()[:-1] == values.size()
    rng = torch.arange(mask.size(-1), out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

def get_costs(dataset, pi, state, mat):
    depots = torch.zeros(pi.size(0), 1).long().to(device)
    _,ind = torch.max(dataset, dim=2)
    bdd = mat.var[state.prev_a.squeeze() * mat.n_c].unsqueeze(1)
    bdd = torch.randn(state.prev_a.size(0), device=device) * bdd
    add = mat.__getd__(ind, state.prev_a, depots, state.lengths).unsqueeze(1)
    bdd = bdd.repeat(1, 2)
    bdd[:,1] = add.squeeze() * 5
    bdd = torch.min(bdd, dim=1)[0]
    bdd = bdd[:, None].repeat(1, 2)
    bdd[:,1] = add.squeeze() * -0.9
    bdd = torch.max(bdd, dim=1)[0]
    return state.lengths.squeeze() + add.squeeze() + bdd.squeeze(), None

class StateTSP(NamedTuple):
    # Fixed input
    loc: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    tot: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property

    def visited(self):
        if self.visited_.dtype == torch.bool:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                first_a=self.first_a[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                tot=self.tot[key],
                cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
            )
        return super(StateTSP, self).__getitem__(key)

    @staticmethod
    def initialize(loc, visited_dtype=torch.bool):

        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        return StateTSP(
            loc=loc,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.bool, device=loc.device
                )
                if visited_dtype == torch.bool
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=None,
            tot = torch.full((batch_size, 1), n_loc-2, device=loc.device, dtype=torch.long),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)


    def addmask(self):
        visited_ = self.visited_.scatter(-1, self.first_a[:, :, None], 1)


        '''
        preva = torch.tensor([self.visited_.size(2)-8], dtype=torch.long, device=self.visited_.device)
        preva = preva[None,:].expand_as(self.prev_a)
        visited_ = visited_.scatter(-1, preva[:, :, None], 1)
        preva = torch.tensor([self.visited_.size(2)-7], dtype=torch.long, device=self.visited_.device)
        preva = preva[None,:].expand_as(self.prev_a)
        visited_ = visited_.scatter(-1, preva[:, :, None], 1)
        preva = torch.tensor([self.visited_.size(2)-6], dtype=torch.long, device=self.visited_.device)
        preva = preva[None,:].expand_as(self.prev_a)
        visited_ = visited_.scatter(-1, preva[:, :, None], 1)
        preva = torch.tensor([self.visited_.size(2)-5], dtype=torch.long, device=self.visited_.device)
        preva = preva[None,:].expand_as(self.prev_a)
        visited_ = visited_.scatter(-1, preva[:, :, None], 1)


        preva = torch.tensor([self.visited_.size(2)-4], dtype=torch.long, device=self.visited_.device)
        preva = preva[None,:].expand_as(self.prev_a)
        visited_ = visited_.scatter(-1, preva[:, :, None], 1)
        preva = torch.tensor([self.visited_.size(2)-3], dtype=torch.long, device=self.visited_.device)
        preva = preva[None,:].expand_as(self.prev_a)
        visited_ = visited_.scatter(-1, preva[:, :, None], 1)

        preva = torch.tensor([self.visited_.size(2)-2], dtype=torch.long, device=self.visited_.device)
        preva = preva[None,:].expand_as(self.prev_a)
        visited_ = visited_.scatter(-1, preva[:, :, None], 1)
        preva = torch.tensor([self.visited_.size(2)-1], dtype=torch.long, device=self.visited_.device)
        preva = preva[None,:].expand_as(self.prev_a)
        visited_ = visited_.scatter(-1, preva[:, :, None], 1)
        '''

        return self._replace(visited_=visited_)        

    def update(self, selected, mat, ind, inp1, inp2):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step

        bdd = mat.var[self.prev_a.squeeze() * mat.n_c + prev_a.squeeze()].unsqueeze(1)
        add = mat.__getd__(ind, self.prev_a, prev_a, self.lengths).unsqueeze(1)
        bdd = torch.randn(prev_a.size(0), 1, device=device) * bdd
        bdd = bdd.repeat(1, 2)
        bdd[:, 1] = add.squeeze() * 5
        bdd = torch.min(bdd, dim=1)[0]
        bdd = bdd[:, None].repeat(1, 2)
        bdd[:, 1] = add.squeeze() * -0.9
        bdd = torch.max(bdd, dim=1)[0]
        lengths = self.lengths + add + bdd[:, None]
        visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        is_ful = visited_[:, 0, 1:visited_.size(-1)].sum(-1).eq(visited_.size(-1) - 1).long()
        id = inp1[:,self.i].repeat(1, 2)
        id[:,1] = 1
        id = id.gather(-1, is_ful[:, None])
        yy = torch.min(visited_.squeeze(), dim=1)[1]
        yy = yy[:,None].expand(yy.size(0), inp2.size(1))
		
        newmask = visited_.squeeze().unsqueeze(2).repeat(1, 1, 3)
        newmask[:,:,0].scatter_(1, self.tot+1, 0)
        newmask[:,:,2].scatter_(1, yy, 1)


        tot = self.tot
        newtot = tot.repeat(1, 3)
        newtot[:,0] = torch.clamp(tot.squeeze() + 1, max=inp2.size(1)-2)
        tot = newtot.gather(1, id)
        idx = id.unsqueeze(2).expand(id.size(0), inp2.size(1), 1)
        visited_ = newmask.gather(2, idx).squeeze().unsqueeze(1)

        return self._replace(prev_a=prev_a, visited_=visited_, lengths=lengths, tot=tot, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.loc.size(-2) - 1

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited_

class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)

class AttentionModel2(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 input_size=4,
                 max_t=12):
        
        super(AttentionModel2, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.n_heads = n_heads
        step_context_dim = 2 * embedding_dim  # Embedding of first and last node
        node_dim = 100
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
        self.input_size = input_size
        self.node_dim = node_dim

        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.mask_embed = nn.Linear(1, embedding_dim)
        self.step_embed = nn.Linear(embedding_dim*2, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        self.embed_static_traffic = nn.Linear(input_size * max_t, embedding_dim)
        self.embed_static = nn.Linear(2 * embedding_dim, embedding_dim)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.xx = torch.tensor([[i for j in range(node_dim)] for i in range(input_size)], device=device).view(1, input_size*node_dim)
        self.yy = torch.tensor([[j for j in range(node_dim)] for i in range(input_size)], device=device).view(1, input_size*node_dim)
    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
    
    def forward(self, mat, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        
        _log_p, pi, state = self._inner(input, mat)

        cost, mask = get_costs(input, pi, state, mat)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        return cost, ll, pi

    def _init_embed(self, x):
        return self.init_embed(x)
    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _inner(self, input, mat):

        

        outputs = []
        sequences = []
        _, ind = torch.max(input, dim=2)

        state = StateTSP.initialize(input)
        state = state.addmask()

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step

        batch_size, input_size = state.ids.size(0), input.size(1)
        node_dim = self.node_dim



        while not (state.all_finished()):
            traffic = mat.__getddd__(ind, self.xx.expand(batch_size, input_size*node_dim), self.yy.expand(batch_size, input_size*node_dim), state.lengths).view(batch_size, input_size, node_dim)
            step_embedding = self.step_embed(torch.cat((self.init_embed(traffic), self.mask_embed(state.get_mask().squeeze().unsqueeze(2).float())), dim=2))
            embeddings, _ = self.embedder(step_embedding)
            
            # Perform decoding steps
            fixed = self._precompute(embeddings)
            
            log_p, mask = self._get_log_p(fixed, state, mat, input)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

            state = state.update(selected, mat, input)

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1), state
    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)
    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
    def _get_log_p(self, fixed, state, mat, input, normalize=True):

        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state, mat, input))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask
    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = F.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)
    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected
    def _get_parallel_step_context(self, embeddings, state, mat, input):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """
        b_s, i_s = embeddings.size(0), embeddings.size(1)
        ss = embeddings.gather(1, torch.cat((state.first_a, state.prev_a), 1)[:, :, None].expand(b_s, 2, embeddings.size(-1)))
        return ss.view(b_s, 1, -1)
        
    def _get_attention_node_data(self, fixed, state):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key
    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 input_size=4,
                 max_t=12):
        
        super(AttentionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.n_heads = n_heads
        step_context_dim = 2 * embedding_dim  # Embedding of first and last node
        node_dim = 100
        self.node_dim = node_dim
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
        self.input_size = input_size

        self.init_embed = nn.Linear(embedding_dim*2, embedding_dim)
        self.mask_embed = nn.Linear(1, embedding_dim)
        self.step_embed = nn.Linear(embedding_dim*2, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        self.embed_static_traffic = nn.Linear(input_size * max_t, embedding_dim)
        self.embed_static = nn.Linear(2 * embedding_dim, embedding_dim)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.xx = torch.tensor([[i for j in range(node_dim)] for i in range(input_size)], device=device).view(1, input_size*node_dim)
        self.yy = torch.tensor([[j for j in range(node_dim)] for i in range(input_size)], device=device).view(1, input_size*node_dim)
    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
    
    def forward(self, mat, inputz, model1, model2, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        
        input = inputz[:,:,:100]
        inp1 = inputz[:,:,100].long()
        inp2 = inputz[:,:,101].long()
        _log_p, pi, state = self._inner(input, mat, model1, model2, inp1, inp2)

        cost, mask = get_costs(input, pi, state, mat)
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        return cost, ll, pi

    def _init_embed(self, x):
        return self.init_embed(x)
    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _inner(self, input, mat, model1, model2, inp1, inp2):

        

        outputs = []
        sequences = []
        _, ind = torch.max(input, dim=2)
        node_dim = self.node_dim

        state = StateTSP.initialize(input)
        state = state.addmask()

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step

        batch_size, input_size = state.ids.size(0), input.size(1)

        with torch.no_grad():
            #graph_embedding[i, j, k] = model1[i, ind[i,j], k]
            graph_embedding = model1.unsqueeze(0).repeat(ind.size(0), 1, 1).gather(1, ind.unsqueeze(2).expand(ind.size(0), ind.size(1), model1.size(1)))

        while not (state.all_finished()):
            traffic = mat.__getddd__(ind, self.xx.expand(batch_size, input_size*node_dim), self.yy.expand(batch_size, input_size*node_dim), state.lengths).view(batch_size, input_size, node_dim)
            graph_embedding = model1.unsqueeze(0).repeat(ind.size(0), 1, 1).gather(1, ind.unsqueeze(2).expand(ind.size(0), ind.size(1), model1.size(1)))
            embedding = model2.step_embed(torch.cat((model2.init_embed(traffic), model2.mask_embed(state.get_mask().squeeze().unsqueeze(2).float())), dim=2))
            step_embedding, _ = model2.embedder(embedding)
            
            embeddings, _ = self.embedder(self._init_embed(torch.cat((graph_embedding, step_embedding),dim=2)))
            # Perform decoding steps
            fixed = self._precompute(embeddings)
            
            log_p, mask = self._get_log_p(fixed, state, mat, ind)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

            state = state.update(selected, mat, ind, inp1, inp2)


            newind = ind[:,:,None].clone().repeat(1, 1, 3)
            teemp = ind.clone()
            teemp.scatter_(1, state.tot.expand_as(inp2), inp2[:,state.i-1:state.i].expand(inp2.size(0),inp2.size(1)))
            newind[:,:,0] = teemp
            idx = inp1[:,state.i-1:state.i].unsqueeze(2).expand(inp1.size(0), inp1.size(1), 1)
            ind = newind.gather(2, idx).squeeze()

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1), state
    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)
    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
    def _get_log_p(self, fixed, state, mat, ind, normalize=True):

        # Compute query = context node embedding
        query = fixed.context_node_projected + \
        self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state, mat, ind))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask().clone()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        tmp = torch.sum(mask.squeeze(), dim=1)
        tmp = tmp.eq(mask.squeeze().size(1)).unsqueeze(1).unsqueeze(1).expand(log_p.size(0), 1, log_p.size(2))
        mask = mask.repeat(1, 2, 1)
        mask[:,1,0] = 0
        mask = mask.gather(1, tmp.long())

        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)

        newlog = log_p.repeat(1, 2, 1)
        newlog[:,1,:] = torch.full((log_p.size(0), log_p.size(2)), math.log(1e-6))
        newlog[:,1,0] = torch.full((log_p.size(0), 1), math.log(1 - log_p.size(2) * 1e-6)).squeeze()
        log_p = newlog.gather(1, tmp.long())
        assert not torch.isnan(log_p).any()

        return log_p, mask
    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = F.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)
    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected
    def _get_parallel_step_context(self, embeddings, state, mat, ind):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """
        b_s, i_s = embeddings.size(0), embeddings.size(1)
        ss = embeddings.gather(1, torch.cat((state.first_a, state.prev_a), 1)[:, :, None].expand(b_s, 2, embeddings.size(-1)))
        return ss.view(b_s, 1, -1)
        
    def _get_attention_node_data(self, fixed, state):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key
    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)