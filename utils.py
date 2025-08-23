import torch
import math
import time
import numpy as np

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        if initial_weights is None:
            initial_weights = torch.zeros_like(self.classification_head.weight)
            torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
        self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
        self.classification_head.bias = torch.nn.Parameter(
            torch.zeros_like(self.classification_head.bias))

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, return_features=False):
        features = self.model.encode_image(images)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        if return_features:
            return logits, features
        return logits

def get_model_from_sd(state_dict, base_model):
    feature_dim = state_dict['classification_head.weight'].shape[1]
    num_classes = state_dict['classification_head.weight'].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    for p in model.parameters():
        p.data = p.data.float()
    model.load_state_dict(state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if device == 'cuda':
        devices = [x for x in range(torch.cuda.device_count())]
        return torch.nn.DataParallel(model,  device_ids=devices)
    return model

def maybe_dictionarize_batch(batch):
    if isinstance(batch, dict):
        return batch
    if len(batch) == 2:
        return {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        return {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')


def eval_model_on_dataset(model, dataloader):
    """
    Run model prediction on the given dataloader, including dataloader of training set , validation set or test set
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        end = time.time()

        for i, batch in enumerate(dataloader):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].to(device), batch['labels'].to(device)
            data_time = time.time() - end
            y = labels
            if 'image_paths' in batch:
                image_paths = batch['image_paths']

            logits = model(inputs)

            if isinstance(logits, list):
                logits = logits[0]

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

            batch_time = time.time() - end
            end = time.time()
            if i % 20 == 0:
                percent_complete = 100.0 * i / len(dataloader)
                print(
                    f"[{percent_complete:.0f}% {i}/{len(dataloader)}]\t"
                    f"Acc: {100 * (correct/n):.2f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                )

        top1 = correct / n
        return top1


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster
