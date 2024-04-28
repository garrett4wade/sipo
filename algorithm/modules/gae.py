import torch


@torch.no_grad()
def masked_normalization(x,
                         mask=None,
                         dim=None,
                         inplace=False,
                         unbiased=False,
                         eps=torch.tensor(1e-5)):
    if not inplace:
        x = x.clone()
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        mask = torch.ones_like(x)
    x = x * mask
    factor = mask.sum(dim=dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return (x - mean) / (var.sqrt() + eps)


@torch.no_grad()
def gae_trace(
        reward,
        value,
        mask,
        gamma=torch.tensor(0.99),
        gae_lambda=torch.tensor(0.95),
        bad_mask=None,
):
    assert mask.shape[0] == value.shape[0] == reward.shape[0]
    assert reward.shape == value.shape
    if bad_mask is None:
        bad_mask = torch.ones_like(mask)
    else:
        assert bad_mask.shape == mask.shape
    episode_length = int(reward.shape[0]) - 1

    delta = reward[:-1] + gamma * value[1:] * mask[1:] - value[:-1]
    gae = torch.zeros_like(reward[0])
    adv = torch.zeros_like(reward)
    m = gamma * gae_lambda * mask[1:]
    step = episode_length - 1
    while step >= 0:
        gae = delta[step] + m[step] * gae
        adv[step] = gae * bad_mask[step + 1]
        step -= 1
    return adv


@torch.no_grad()
def rspo_gae_trace(
        reward,
        value,
        mask,
        num_refs,
        gamma=torch.tensor(0.99),
        gae_lambda=torch.tensor(0.95),
        bad_mask=None,
):
    assert mask.shape[0] == value.shape[0] == reward.shape[0]
    assert reward.shape[-1] == 3 * num_refs + 1, reward.shape
    assert value.shape[-1] == 2 * num_refs + 1, value.shape
    assert reward.shape[:-1] == value.shape[:-1], (reward.shape, value.shape)
    if bad_mask is None:
        bad_mask = torch.ones_like(mask)
    else:
        assert bad_mask.shape == mask.shape
    episode_length = int(reward.shape[0]) - 1

    delta = reward[:-1, ..., :1 +
                   num_refs * 2] + gamma * value[1:] * mask[1:] - value[:-1]
    gae = torch.zeros_like(reward[0, ..., :1 + num_refs * 2])
    adv = torch.zeros_like(reward[..., :1 + num_refs * 2])
    m = gamma * gae_lambda * mask[1:]
    step = episode_length - 1

    rspo_return = torch.zeros_like(reward[..., 1 + num_refs * 2:])
    while step >= 0:
        gae = delta[step] + m[step] * gae
        adv[step] = gae * bad_mask[step + 1]
        rspo_return[step] = reward[
            step, ...,
            1 + num_refs * 2:] + gamma * mask[step + 1] * rspo_return[step + 1]
        step -= 1
    return adv, rspo_return