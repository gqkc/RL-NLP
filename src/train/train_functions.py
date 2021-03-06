import time
import torch
import torch.nn.functional as F


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history.
    so that each hidden_state h_t is detached from the backprop graph once used. """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train_one_epoch(model, train_generator, optimizer, criterion, device, args, print_interval=10):
    model.train()  # Turns on train mode which enables dropout.
    total_loss = 0.
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(train_generator):
        inputs = inputs.to(device)
        targets = targets.view(targets.size(1) * targets.size(0)).to(device)  # targets (S*B)
        model.zero_grad()  # TODO: is there a difference between model.zero_grad() and optimizer.zero_grad()
        output, hidden = model(inputs)  # output (S * B, V), hidden (num_layers,B,1)
        loss = criterion(output, targets)
        loss.backward()
        # clip grad norm:
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        total_loss += loss.item()
        # print loss every number of batches
        if (batch + 1) % print_interval == 0:
            print('loss for batch {}: {:5.3f}'.format(batch + 1, total_loss / (batch + 1)))
            print('time for {} training steps: {:5.2f}'.format(print_interval, time.time() - start_time))

    curr_loss = total_loss / (batch + 1)
    elapsed = time.time() - start_time

    return curr_loss, elapsed

def train_one_epoch_policy(model, train_generator, optimizer, criterion, device, args, print_interval=10):
    model.train()  # Turns on train mode which enables dropout.
    total_loss = 0.
    start_time = time.time()
    for batch, ((inputs, targets), feats, _) in enumerate(train_generator):
        inputs, feats = inputs.to(device), feats.to(device)
        targets = targets.view(targets.size(1) * targets.size(0)).to(device)  # targets (S*B)
        model.zero_grad()  # TODO: is there a difference between model.zero_grad() and optimizer.zero_grad()
        logits, hidden, _ = model(inputs, feats)  # output (S * B, V), hidden (num_layers,B,1)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = criterion(log_probs, targets)
        loss.backward()
        # clip grad norm:
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        total_loss += loss.item()
        # print loss every number of batches
        if (batch + 1) % print_interval == 0:
            print('loss for batch {}: {:5.3f}'.format(batch + 1, total_loss / (batch + 1)))
            print('time for {} training steps: {:5.2f}'.format(print_interval, time.time() - start_time))

    curr_loss = total_loss / (batch + 1)
    elapsed = time.time() - start_time

    return curr_loss, elapsed


def evaluate(model, val_generator, criterion, device):
    model.eval()  # turn on evaluation mode which disables dropout.
    total_loss = 0.
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(val_generator):
            inputs = inputs.to(device)
            targets = targets.view(targets.size(1) * targets.size(0)).to(device)
            output, hidden = model(inputs)
            total_loss += criterion(output, targets).item()

    return total_loss / (batch + 1)

def evaluate_policy(model, val_generator, criterion, device):
    model.eval()  # turn on evaluation mode which disables dropout.
    total_loss = 0.
    with torch.no_grad():
        for batch, ((inputs, targets), feats, _) in enumerate(val_generator):
            inputs, feats = inputs.to(device), feats.to(device)
            targets = targets.view(targets.size(1) * targets.size(0)).to(device)
            logits, hidden, _ = model(inputs, feats)
            log_probs = F.log_softmax(logits, dim=-1)
            total_loss += criterion(log_probs, targets).item()

    return total_loss / (batch + 1)
