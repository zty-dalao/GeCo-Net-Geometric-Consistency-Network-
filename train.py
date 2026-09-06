import torch
from models.model import model
from data.Dataset import CBCTDataset
from util.train_args import parse_args
from trainer import trainer

if __name__ == '__main__':
    args,conf = parse_args()
    device = args.device
    ## dataset
    train_dataset = CBCTDataset(args, stage="train")
    val_dataset = CBCTDataset(args, stage="val")
    test_dataset = CBCTDataset(args, stage="test")
    visual_dataset = CBCTDataset(args, stage="visual")

    ## dataloader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle = True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle = True,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle = True,
    )
    visual_data_loader = torch.utils.data.DataLoader(
        visual_dataset,
        batch_size=args.batch_size,
        shuffle = False,
    )

    ## model
    G_render = model(
        model_conf=conf['model'],
        device=device,
        query_chunk_size=args.query_chunk_size,
        use_query_checkpoint=not args.disable_query_checkpoint,
        use_adapter=args.use_adapter,
        adapter_hidden_channels=args.adapter_hidden_channels,
    )
    if args.pretrained_backbone is not None and not args.resume:
        backbone_checkpoint = torch.load(args.pretrained_backbone, map_location="cpu")
        source_state = backbone_checkpoint.get("G_render", backbone_checkpoint)
        backbone_state = {
            key: value
            for key, value in source_state.items()
            if key.startswith("encoder.") or key.startswith("aggregator.")
        }
        if not backbone_state:
            raise KeyError(
                f"Main checkpoint {args.pretrained_backbone!r} has no encoder/aggregator weights."
            )
        incompatible = G_render.load_state_dict(backbone_state, strict=False)
        unexpected = [
            key for key in incompatible.unexpected_keys
            if key.startswith("encoder.") or key.startswith("aggregator.")
        ]
        if unexpected:
            raise RuntimeError(
                "Unexpected encoder/aggregator keys while loading pretrained backbone: "
                + ", ".join(unexpected)
            )
        print(
            f"Loaded {len(backbone_state)} encoder/aggregator tensors from: "
            f"{args.pretrained_backbone}"
        )
        del backbone_checkpoint, source_state, backbone_state
    if args.pretrained_decoder is not None and not args.resume:
        # Keep duplicated model/optimizer entries in the pretraining checkpoint
        # off the GPU; load_state_dict copies only decoder tensors to the model.
        pretrained = torch.load(args.pretrained_decoder, map_location="cpu")
        if "decoder" not in pretrained:
            raise KeyError(
                f"Pretraining checkpoint {args.pretrained_decoder!r} has no 'decoder' key."
            )
        G_render.decoder.load_state_dict(pretrained["decoder"], strict=True)
        print(f"Loaded pretrained decoder: {args.pretrained_decoder}")
        del pretrained
    net_trainer = trainer(G_render,train_data_loader,val_data_loader,
    test_data_loader,visual_data_loader,args,conf,device)
    net_trainer.start()
