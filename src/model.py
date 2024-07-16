import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionModule(nn.Module):
    def __init__(self, num_heads, mlp_dim=None, num_attention_layers=1, dropout=0.0, act=nn.Tanh(), sigmoid=True):
        # if mlp_dim == 0 the similarity is calculated by appliyng sigmoid to the first element of the first output token of the attention layer
        super().__init__()
        embed_dim = 512
        
        self.cls_linear = nn.Linear(1, embed_dim) # self.cls = nn.Embedding(1, embed_dim)
        self.attention_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True) for _ in range(num_attention_layers)])
        self.linear_layer1 = nn.Linear(embed_dim, mlp_dim) if mlp_dim is not None else None
        self.linear_layer2 = nn.Linear(mlp_dim, 1) if mlp_dim is not None else None
        self.act = act
        self.sigmoid = sigmoid
        
    
    @classmethod
    def from_config(cls, config):
        model = cls(
            num_heads=config['num_heads'],
            mlp_dim=config.get('mlp_dim', None),
            num_attention_layers=config.get('num_attention_layers', 1),
            dropout=config.get('dropout', 0)
        )
        if 'initial_weights' in config and config['initial_weights'] is not None:
            model.load_state_dict(torch.load(config['initial_weights'], 'cpu'))
        return model

    def forward(self, visual_embedding, textual_embedding, ret_similarity_matrix=False, fgovd=False):
        dim0 = visual_embedding.shape[0] # batch size
        dim1 = visual_embedding.shape[0] if not fgovd else textual_embedding.shape[1] # batch size if fgovd else number of negatives + 1
        
        if ret_similarity_matrix:
            # in case we have to build a similarity matrix, we create couple of text_embedding visual_embedding
            visual_embedding = visual_embedding.expand(dim1, dim0, 512).permute(1, 0, 2).reshape(dim0 * dim1, 512)# (dim0 * dim1) x 512
            textual_embedding = textual_embedding.expand(dim0, dim1, 512).reshape(dim0 * dim1, 512) # (dim0 * dim1) x 512
            input_dim = dim0 * dim1
        else:
            input_dim = dim0
        cls = self.cls_linear(torch.ones(input_dim, 1, dtype=torch.float32).to(visual_embedding.device)) # cls = self.cls.weight.expand(visual_embedding.shape[0], -1) # expanding CLS token along batch dimension
        
        x = torch.stack((cls, visual_embedding, textual_embedding), dim=1) # BS x 3 x 512 if not ret_similarity_matrix else (dim0 * dim1) x 3 x 512
        for att_layer in self.attention_layers:
            x, _ = att_layer(x, x, x)
        if self.linear_layer1:
            # act = nn.Tanh()
            x = self.linear_layer1(x[:, 0])
            x = self.act(x)
            x = self.linear_layer2(x)
            if not self.sigmoid:
                x = x.squeeze(1)
            if self.sigmoid:
                x = torch.sigmoid(x).squeeze(1) # BS x 1 if not ret_similarity_matrix else dim0 * dim1 x 1
        else:
            x = x[:, 0, 0]
            if self.sigmoid:
                x = torch.sigmoid(x) # BS x 1 if not ret_similarity_matrix else dim0 * dim1 x 1

        # if we want to return the similarity matrix we have to view the matrix as dim0 x dim0
        if ret_similarity_matrix:
            x = x.view(dim0, dim1) # dim0 x dim1
        
        return x
    
    
    def __len__(self):
        return sum(p.numel() for p in self.parameters())


class MLPs(nn.Module):
    def __init__(self, mlp_dims=[512], keep_embeds=[False, False], act=nn.Tanh(), rescaling=False, sigmoid=True, cosine=False):
        # mlp_dims list of mlp dimensions
        super().__init__()
        embedding_dim = 512
        last_dim = embedding_dim
        n_embeddings = len(keep_embeds)
        
        linear_layers = [[] for _ in range(n_embeddings)]
        for mlp_dim in mlp_dims:
            for i, keep_embed in zip(range(n_embeddings), keep_embeds):
                if not keep_embed:
                    linear_layers[i].append(nn.Linear(last_dim, mlp_dim))
            last_dim = mlp_dim
            
        self.linear_layers = []
        for layers, keep_embed in zip(linear_layers, keep_embeds):
            if not keep_embed:
                layers.append(nn.Linear(last_dim, embedding_dim))
                self.linear_layers.append(nn.ModuleList(layers))
            else:
                self.linear_layers.append(None)
        self.linear_layers = nn.ModuleList(self.linear_layers) 
        self.act = act
        
        # if rescaling we add a 512 tensor to-learn (weight of a linear layer) to perforn element wise multiplication with text embeddings
        if rescaling:
            self.rescaling = nn.Linear(embedding_dim, 1, bias=False)
        else:
            self.rescaling = None
            
        self.sigmoid = sigmoid
        self.cosine = cosine
    
    @classmethod
    def from_config(cls, config):
        no_act = config.get('no_act', False)
        act = nn.Tanh() if not no_act else None
        model = cls(config.get('mlp_dims', [512]),
                    config.get('keep_embeds', [False, False]),
                    act=act,
                    rescaling=config.get('rescaling', False),
                    sigmoid=config.get('sigmoid', True),
                    cosine=config.get('cosine', False))
        if 'initial_weights' in config and config['initial_weights'] is not None:
            model.load_state_dict(torch.load(config['initial_weights'], 'cpu'))
        return model
    
    def forward(self, visual_embedding, textual_embedding, ret_similarity_matrix=False, fgovd=False, ret_embeds=False):
        embeds = [visual_embedding.float(), textual_embedding.float()]
        out_embeds = []
        for i, (embed, linear_layers) in enumerate(zip(embeds, self.linear_layers)):
            x = embed
            if linear_layers is not None:
                for linear_layer in linear_layers:
                    x = linear_layer(x)
                    if self.act is not None:
                        x = self.act(x)
            if self.rescaling is not None and i == 1: # rescale only text embedding
                x = self.rescaling.weight * x
            out_embeds.append(x)
        
        if self.cosine:
            out_embeds[0] = F.normalize(out_embeds[0], p=2, dim=1)
            out_embeds[1] = F.normalize(out_embeds[1], p=2, dim=1)
        if ret_embeds:
            return out_embeds[0], out_embeds[1]
        if not fgovd:
            x = out_embeds[0] @ out_embeds[1].transpose(1, 0)
            if not ret_similarity_matrix:
                x = x[torch.eye(len(x)) > 0.5] # only diagonal elements
        else:
            x = out_embeds[0].unsqueeze(1) @ out_embeds[1].transpose(2, 1)
            x = x.squeeze(1)
        
        if self.sigmoid:
            x = nn.Sigmoid()(x)
        return x
    
    def __len__(self):
        return sum(p.numel() for p in self.parameters())    
    

def main():
    pass

if __name__ == '__main__':
    main()