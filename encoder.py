import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Params:
    SIZE = (224,224)
    PATCH_SIZE = 16
    NR_PATCHES = int((224/16)**2)
    EMBEDDING = PATCH_SIZE**2*3
    NUN_HEADS = 1
    NR_LAYERS = 12
    VOCAB_SIZE = 503
    HIDDEN = 768

class PatchEmbeding(nn.Module):
    ## (3,224,224) - size
    def __init__(self, size, patch_size, embedding_size):
        super().__init__()
        self.size = size
        self.patch_size = patch_size
        self.n_patches = int((224 // self.patch_size) ** 2)
        self.embedding_size = embedding_size

        self.patch = nn.Conv2d(
            in_channels=3,
            out_channels=embedding_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0
        )
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, image):
        assert image.shape[-1] % self.patch.kernel_size[0] == 0, "input image must be divisible by the patches"
        patches = self.patch(image)
        patches = self.flatten(patches)

        return patches.permute(0, 2, 1)


class MLP(nn.Module):
    def __init__(self, embedding_size, dropout=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.dropout = dropout

        self.fc1 = nn.Linear(self.embedding_size, self.embedding_size * 4)
        self.fc2 = nn.Linear(self.embedding_size * 4, self.embedding_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class MHABlock(nn.Module):
    def __init__(self, embedding_size, n_heads=2, dropout=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.dropout = dropout

        self.attn = nn.MultiheadAttention(self.embedding_size, self.n_heads, bias=True)

    def forward(self, query, key, value, mask=None):
        output, _ = self.attn(query, key, value, attn_mask=mask)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        self.mlp = MLP(self.embedding_size)
        self.attn = MHABlock(self.embedding_size, self.num_heads)

        self.norm2 = nn.LayerNorm(self.embedding_size)
        self.norm1 = nn.LayerNorm(self.embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        x = self.dropout(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size, num_heads, nr_layers):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.nr_layers = nr_layers
        self.encoder = nn.ModuleList(
            [TransformerBlock(self.embedding_size, self.num_heads) for _ in range(self.nr_layers)])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        return x


class ViT_Model(nn.Module):
    def __init__(self, img_size, patch_size, nr_patches, embedding_size, num_heads, nr_layers, ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.nr_patches = nr_patches
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.nr_layers = nr_layers

        # PATCHING + POSITIONAL EMBEDDINGS + CLASS EMBEDDINGS ?
        self.patcher = PatchEmbeding(self.img_size, self.patch_size, self.embedding_size)
        self.positional_embeddings = nn.Parameter(torch.randn(1, self.nr_patches, self.embedding_size))
        # self.class_embeddings = nn.Parameter(torch.randn(1,1,self.embedding_size))

        self.transformer_encoder = TransformerEncoder(self.embedding_size, self.num_heads, self.nr_layers)
        self.dropout = nn.Dropout(0.1)

        self.feature_extractor = MLP(embedding_size=self.embedding_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patcher(x)
        b, n, _ = x.shape
        x += self.positional_embeddings
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        logits = x[:, 0]
        logits = self.feature_extractor(logits)

        return logits


