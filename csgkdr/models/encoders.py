import torch

from torch import nn
from torch_geometric.nn.models import GIN
from torch_geometric.nn import GATConv, SAGPooling, global_mean_pool


class GINEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        out_channels,
        dropout,
        norm=None,
        emb_style="none",
        num_nodes=None,
        emb_size=128,
        use_attention=True,
        use_hierarchical_pooling=True,
    ):
        super().__init__()
        self.emb_style = emb_style
        self.num_nodes = num_nodes
        self.use_attention = use_attention
        self.use_hierarchical_pooling = use_hierarchical_pooling

        if emb_style not in ["none", "concat", "replace"]:
            raise ValueError(f"Invalid emb_style: {emb_style}")

        if emb_style != "none":
            self.emb = nn.Embedding(num_nodes, emb_size)

        if emb_style == "concat":
            in_channels += emb_size
        elif emb_style == "replace":
            in_channels = emb_size

        # base GIN model
        self.gin = GIN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
        )
        
        # attention layers
        if self.use_attention:
            self.attention_layers = nn.ModuleList([
                GATConv(out_channels, out_channels // 4, heads=4, dropout=dropout)
                for _ in range(2)  # two GAT layers
            ])
            self.node_attention = nn.Sequential(
                nn.Linear(out_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, 1)
            )
        
        # hierarchical pooling
        if self.use_hierarchical_pooling:
            self.pool1 = SAGPooling(out_channels, ratio=0.8)
            self.pool2 = SAGPooling(out_channels, ratio=0.6)
        self.projection = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, batch_graph):
        if self.emb_style != "none":
            x = self._add_emb(batch_graph.x, batch_graph.batch_size)
        else:
            x = batch_graph.x

        edge_index = batch_graph.edge_index
        batch = getattr(batch_graph, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.gin(x, edge_index)
        if self.use_attention:
            for attention_layer in self.attention_layers:
                attention_x = attention_layer(x, edge_index)
                x = x + attention_x  # residual connection
            node_weights = self.node_attention(x).squeeze(-1)
            node_weights = torch.sigmoid(node_weights)
            x = x * node_weights.unsqueeze(-1) 
    
        if self.use_hierarchical_pooling:
            x1, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
            global_x1 = global_mean_pool(x1, batch)
            x2, edge_index, _, batch, _, _ = self.pool2(x1, edge_index, None, batch)
            global_x2 = global_mean_pool(x2, batch)
            x = global_x1 + global_x2
        else:
            x = x.reshape(batch_graph.batch_size, -1, x.shape[1])
            x = x.mean(dim=1)
        x = self.projection(x)
        return x.reshape(-1)

    def _add_emb(self, x, batch_size):
        nids = torch.arange(self.num_nodes).repeat(batch_size).to(x.device)
        emb = self.emb(nids)

        if self.emb_style == "concat":
            x = torch.cat([x, emb], dim=1)
        elif self.emb_style == "replace":
            x = emb

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class SFCNEncoder(nn.Module):
    def __init__(
        self,
        channel_number=[32, 64, 128, 256, 256, 64],
        use_attention=True, 
        use_depthwise_conv=True, 
    ):
        super().__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        self.use_attention = use_attention
        self.use_depthwise_conv = use_depthwise_conv
        
        # attention modules
        if self.use_attention:
            self.channel_attentions = nn.ModuleList()
            self.spatial_attentions = nn.ModuleList()
        
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i - 1]
            out_channel = channel_number[i]
            
            if i < n_layer - 1:
                if self.use_depthwise_conv and i > 0: 
                    self.feature_extractor.add_module(
                        f"depthwise_conv_{i}",
                        self.depthwise_separable_conv_layer(
                            in_channel, out_channel, maxpool=True, kernel_size=3, padding=1
                        )
                    )
                else:
                    self.feature_extractor.add_module(
                        f"conv_{i}",
                        self.conv_layer(
                            in_channel, out_channel, maxpool=True, kernel_size=3, padding=1
                        )
                    )
                
                if self.use_attention:
                    self.channel_attentions.append(ChannelAttention(out_channel))
                    self.spatial_attentions.append(SpatialAttention())
            else:
                self.feature_extractor.add_module(
                    f"conv_{i}",
                    self.conv_layer(
                        in_channel, out_channel, maxpool=False, kernel_size=1, padding=0
                    )
                )

        avg_shape = [5, 6, 5]
        self.feature_extractor.add_module("avgpool", nn.AvgPool3d(avg_shape))
        self.projection = nn.Sequential(
            nn.Linear(channel_number[-1], channel_number[-1]),
            nn.BatchNorm1d(channel_number[-1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        bsize = x.shape[0]
        features = []
        if self.use_attention:
            modules = list(self.feature_extractor.children())
            attention_idx = 0
            current_x = x
            for i, module in enumerate(modules):
                # convolution
                current_x = module(current_x)
                if isinstance(module, nn.Sequential) and attention_idx < len(self.channel_attentions):
                    channel_attn = self.channel_attentions[attention_idx]
                    channel_weight = channel_attn(current_x)
                    current_x = current_x * channel_weight
                    spatial_attn = self.spatial_attentions[attention_idx]
                    spatial_weight = spatial_attn(current_x)
                    current_x = current_x * spatial_weight
                    attention_idx += 1
                features.append(current_x)
            x = features[-1]
        else:
            x = self.feature_extractor(x)
        x = x.reshape(bsize, -1)
        # projection
        x = self.projection(x)
        return x

    @staticmethod
    def conv_layer(
        in_channel,
        out_channel,
        maxpool=True,
        kernel_size=3,
        padding=0,
        maxpool_stride=2,
    ):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channel, out_channel, padding=padding, kernel_size=kernel_size
                ),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channel, out_channel, padding=padding, kernel_size=kernel_size
                ),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
            )
        return layer
        
    @staticmethod
    def depthwise_separable_conv_layer(
        in_channel,
        out_channel,
        maxpool=True,
        kernel_size=3,
        padding=0,
        maxpool_stride=2,
    ):
        # depthwise-separable convolution = depthwise + pointwise
        depth_conv = nn.Conv3d(
            in_channel, in_channel, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=in_channel,
            bias=False
        )
        point_conv = nn.Conv3d(
            in_channel, out_channel, 
            kernel_size=1,
            bias=False
        )
        
        if maxpool is True:
            layer = nn.Sequential(
                depth_conv,
                point_conv,
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                depth_conv,
                point_conv,
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
            )
        return layer