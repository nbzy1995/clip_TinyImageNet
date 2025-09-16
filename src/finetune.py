import argparse
import os
import torch
import clip
import os
from tqdm import tqdm
import time

from timm.data.transforms_factory import transforms_imagenet_train

from dataset.tiny_imagenet import TinyImageNet
from utils import ModelWrapper, maybe_dictionarize_batch, cosine_lr
from zeroshot import zeroshot_classifier
from openai_imagenet_template import openai_imagenet_template


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=".",
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=".",
        help="Where to save model checkpoints.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--custom-template", action="store_true", default=False,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--warmup-length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--model",
        default='ViT-B/32',
        help='Model to use -- you can try another like ViT-L/14'
    )
    parser.add_argument(
        "--name",
        default='finetune_clip',
        help='Filename for the checkpoints.'
    )
    parser.add_argument(
        "--timm-aug", action="store_true", default=False,
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.custom_template:
        template = [lambda x : f"a photo of a {x}."]
    else:
        template = openai_imagenet_template

    base_model, preprocess = clip.load(args.model, DEVICE, jit=False)

    if args.timm_aug:
        train_preprocess = transforms_imagenet_train(
                img_size=base_model.visual.input_resolution,
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
    else:
        train_preprocess = preprocess
    
    dset = TinyImageNet(eval_preprocess = preprocess, train_preprocess= train_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)

    clf = zeroshot_classifier(base_model, dset.classnames, template, DEVICE)
    NUM_CLASSES = len(dset.classnames)
    feature_dim = base_model.visual.output_dim

    model = ModelWrapper(base_model, feature_dim, NUM_CLASSES, normalize=True, initial_weights=clf)
    for p in model.parameters():
        p.data = p.data.float()

    model = model.to(DEVICE)
    if DEVICE == 'cuda':
        devices = [x for x in range(torch.cuda.device_count())]
        model = torch.nn.DataParallel(model,  device_ids=devices)

    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.wd)

    num_batches = len(dset.train_loader)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    loss_fn = torch.nn.CrossEntropyLoss()

    model_path = os.path.join(args.model_location, f'{args.name}_0.pt')
    print('Saving model to', model_path)
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(model_state, model_path)

    for epoch in range(args.epochs):
        # Train
        model.train()
        end = time.time()
        for i, batch in enumerate(dset.train_loader):
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)
            data_time = time.time() - end

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            if i % 20 == 0:
                percent_complete = 100.0 * i / len(dset.train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        # #Evaluate
        model.eval()
        with torch.no_grad():
            print('*'*80)
            print('Starting eval on validation split')
            correct, count = 0.0, 0.0
            pbar = tqdm(dset.val_loader)
            for batch in pbar:
                batch = maybe_dictionarize_batch(batch)
                inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)

                logits = model(inputs)

                loss = loss_fn(logits, labels)

                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                count += len(logits)
                pbar.set_description(
                    f"Val loss: {loss.item():.4f}   Acc: {100*correct/count:.2f}")
            top1 = correct / count
        print(f'Val acc at epoch {epoch}: {100*top1:.2f}')

        model_path = os.path.join(args.model_location, f'{args.name}_{epoch + 1}.pt')
        print('Saving model to', model_path)
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(model_state, model_path)

