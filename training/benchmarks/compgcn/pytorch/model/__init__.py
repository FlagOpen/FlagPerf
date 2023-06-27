import torch

from .models import CompGCN_ConvE
from dataloaders.dataloader import Data
from utils.utils import in_out_norm


def create_model(args, data:Data):
    graph = data.g.to(args.device)
    num_rel = torch.max(graph.edata["etype"]).item() + 1

    # Compute in/out edge norms and store in edata
    graph = in_out_norm(graph)

    compgcn_model = CompGCN_ConvE(
        num_bases=args.num_bases,
        num_rel=num_rel,
        num_ent=graph.num_nodes(),
        in_dim=args.init_dim,
        layer_size=args.layer_size,
        comp_fn=args.opn,
        batchnorm=True,
        dropout=args.dropout,
        layer_dropout=args.layer_dropout,
        num_filt=args.num_filt,
        hid_drop=args.hid_drop,
        feat_drop=args.feat_drop,
        ker_sz=args.ker_sz,
        k_w=args.k_w,
        k_h=args.k_h,
    )

    return compgcn_model