# --- Stratified sampling and group helpers ---
import hashlib
import math
import operator
import os
import sys
from collections import Counter, OrderedDict, defaultdict
from numbers import Number

import numpy as np
import torch


def is_running_in_vscode():
    """Detect if we're running in VS Code environment"""
    # Check for VS Code specific environment variables
    vscode_indicators = [
        "VSCODE_PID",
        "VSCODE_IPC_HOOK",
        "VSCODE_INJECTION",
        "TERM_PROGRAM",
    ]

    for indicator in vscode_indicators:
        if indicator in os.environ:
            if indicator == "TERM_PROGRAM" and os.environ[indicator] == "vscode":
                return True
            elif indicator != "TERM_PROGRAM":
                return True

    # Check if we're in an interactive Jupyter environment (common in VS Code)
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            # Additional check for VS Code Jupyter extension
            if hasattr(get_ipython(), "kernel"):
                return True
    except ImportError:
        pass

    # Check for Slurm environment variables (if present, likely not VS Code)
    slurm_indicators = ["SLURM_JOB_ID", "SLURM_PROCID", "SLURM_TASK_PID"]
    if any(indicator in os.environ for indicator in slurm_indicators):
        return False

    return False


def build_group_index(ds):
    groups = defaultdict(list)
    ys = getattr(ds, "labels", None)
    attrs = getattr(ds, "attributes", None)
    if (
        ys is not None
        and attrs is not None
        and len(ys) == len(ds)
        and len(attrs) == len(ds)
    ):
        for i in range(len(ds)):
            y_i = int(ys[i])
            a_i = int(attrs[i])
            groups[(y_i, a_i)].append(i)
    else:
        for i in range(len(ds)):
            try:
                _, _, y, a, _ = ds[i]
                groups[(int(y), int(a))].append(i)
            except Exception:
                pass
    return groups


def w(groups, size, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    total = sum(len(ixs) for ixs in groups.values())
    if total == 0:
        return []
    desired = {}
    fracs = {}
    for g, ixs in groups.items():
        c = size * (len(ixs) / total)
        desired[g] = int(np.floor(c))
        fracs[g] = c - desired[g]
    rem = size - sum(desired.values())
    if rem > 0:
        for g in sorted(groups.keys(), key=lambda g: fracs[g], reverse=True)[:rem]:
            desired[g] += 1
    selected = []
    for g, k in desired.items():
        if k <= 0:
            continue
        ixs = groups[g]
        if k >= len(ixs):
            selected.extend(ixs)
        else:
            chosen = rng.choice(ixs, size=k, replace=False).tolist()
            selected.extend(chosen)
    import random

    random.shuffle(selected)
    return selected


def group_counts(ds, name="dataset", max_items=None):
    attr_counter = Counter()
    label_counter = Counter()
    pair_counter = Counter()
    n = len(ds) if max_items is None else min(len(ds), max_items)
    for k in range(n):
        try:
            idx, x, y, a = ds[k]
        except Exception as e:
            print(f"[{name}] Could not read item {k}: {e}")
            break
        y_i = int(y)
        a_i = int(a)
        attr_counter[a_i] += 1
        label_counter[y_i] += 1
        pair_counter[(y_i, a_i)] += 1
    print(
        f"[{name}] size={len(ds)} | unique attrs={sorted(attr_counter.keys())} | unique labels={sorted(label_counter.keys())}"
    )
    print(f"[{name}] attr counts: {dict(sorted(attr_counter.items()))}")
    print(f"[{name}] label counts: {dict(sorted(label_counter.items()))}")
    print(f"[{name}] (label, attr) counts: {dict(sorted(pair_counter.items()))}")
    # Expected coverage checks
    num_attrs = getattr(ds, "num_attributes", None)
    num_labels = getattr(ds, "num_labels", None)
    if num_attrs is not None:
        missing_attrs = [a for a in range(int(num_attrs)) if a not in attr_counter]
        if missing_attrs:
            print(f"WARNING: Missing attrs in {name}: {missing_attrs}")
    if num_labels is not None:
        missing_labels = [y for y in range(int(num_labels)) if y not in label_counter]
        if missing_labels:
            print(f"WARNING: Missing labels in {name}: {missing_labels}")
    return attr_counter, label_counter, pair_counter


def print_error_summary(ERROR_LOG):
    """Print summary of all errors encountered"""
    print("\n=== Error Summary ===")
    total_errors = 0
    for error_type, errors in ERROR_LOG.items():
        if errors:
            print(f"\n{error_type.replace('_', ' ').title()}:")
            for context, error_list in errors.items():
                print(f"  {context}: {len(error_list)} errors")
                total_errors += len(error_list)
    if total_errors == 0:
        print("No errors encountered!")
    else:
        print(f"\nTotal errors: {total_errors}")


def prepare_folders(args):
    folders_util = [
        args.output_dir,
        os.path.join(args.output_dir, args.output_folder_name, args.store_name),
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f"===> Creating folder: {folder}")
            os.makedirs(folder)


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        (
            torch.cat(tuple([t.view(-1) for t in dict_1_values]))
            - torch.cat(tuple([t.view(-1) for t in dict_2_values]))
        )
        .pow(2)
        .mean()
    )


class MovingAverage:
    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data


def count_samples_per_class(targets, num_labels):
    counts = Counter()
    for y in targets:
        counts[int(y)] += 1
    return [counts[i] if counts[i] else np.inf for i in range(num_labels)]


def make_balanced_weights_per_sample(targets):
    counts = Counter()
    classes = []
    for y in targets:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)
    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(targets))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def pdb():
    sys.stdout = sys.__stdout__
    import pdb

    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def print_separator():
    print("=" * 80)


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.4f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)


def safe_load(parsed):
    # certain metrics (e.g., AUROC) sometimes saved as a 1-element list
    if isinstance(parsed, list):
        return parsed[0]
    else:
        return parsed


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


# Helper function to safely extract loss value
def safe_extract_loss(loss_dict, default_value=float("nan")):
    """Safely extract loss value from algorithm output"""
    try:
        if isinstance(loss_dict, dict):
            if "loss" in loss_dict:
                loss_val = loss_dict["loss"]
            elif len(loss_dict) > 0:
                # Try to get the first numeric value from the dict
                loss_val = next(iter(loss_dict.values()))
            else:
                return default_value
        else:
            loss_val = loss_dict

        # Convert to float and check for validity
        loss_float = float(loss_val)
        return loss_float if np.isfinite(loss_float) else default_value
    except (TypeError, ValueError, AttributeError):
        return default_value


# Helper function to safely compute metrics
def safe_mean(values, default_value=float("nan")):
    """Safely compute mean, handling empty lists and NaN values"""
    try:
        if not values:
            return default_value
        valid_values = [v for v in values if np.isfinite(v)]
        return float(np.mean(valid_values)) if valid_values else default_value
    except (TypeError, ValueError):
        return default_value


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of dataset corresponding to a random split of the given dataset,
    with n data points in the first dataset and the rest in the last using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def mixup_data(x, y, alpha=1.0, device="cpu"):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def accuracy(network, loader, device):
    num_labels = loader.dataset.num_labels
    num_attributes = loader.dataset.num_attributes
    corrects = torch.zeros(num_attributes * num_labels)
    totals = torch.zeros(num_attributes * num_labels)

    network.eval()
    with torch.no_grad():
        for _, x, y, a in loader:
            p = network.predict(x.to(device))
            p = (
                (p > 0).cpu().eq(y).float()
                if p.squeeze().ndim == 1
                else p.argmax(1).cpu().eq(y).float()
            )
            groups = num_attributes * y + a
            for g in groups.unique():
                corrects[g] += p[groups == g].sum()
                totals[g] += (groups == g).sum()
        corrects, totals = corrects.tolist(), totals.tolist()

        total_acc = sum(corrects) / sum(totals)
        group_acc = [c / t if t > 0 else np.inf for c, t in zip(corrects, totals)]
    network.train()

    return total_acc, group_acc


def adjust_learning_rate(optimizer, lr, step, total_steps, schedule, cos=False):
    """Decay the learning rate based on schedule"""
    if cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if step >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def safe_float_env(var, default):
    """Safely get a float environment variable, with fallback to default."""
    val = os.environ.get(var, "")
    try:
        return float(val) if val else default
    except ValueError:
        return default


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers, seed=None):
        super().__init__()

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, replacement=True, num_samples=len(dataset), generator=generator
            )
        else:
            sampler = torch.utils.data.RandomSampler(
                dataset, replacement=True, generator=generator
            )

        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=False
        )

        self.dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, batch_sampler=batch_sampler
        )
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        while True:
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)
                continue  # retry loop instead of yielding here
            yield batch

    def __len__(self):
        return len(self.dataloader)
