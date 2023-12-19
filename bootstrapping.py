import torch

def get_diffs(model, dataloader):
    model.eval()
    with torch.no_grad():
        diffs = []
        for (data,) in dataloader:
            data = data.to(model.device)
            output = model(data)
            output_flip = model(data.flip(-1, ))
            diff = (output - output_flip).cpu().detach()
            diffs.append(diff)
        diffs = torch.cat(diffs, dim=0)
        print(diffs.shape)
        return diffs

def get_bootstrap_means(model, dataloader, num_bootstraps=10000):
    diffs = get_diffs(model, dataloader)

    bootstrap_means = []
    for _ in range(num_bootstraps):
        bootstrap_sample = torch.randint(0, diffs.shape[0], (diffs.shape[0],), dtype=torch.long)
        resampled_data = diffs[bootstrap_sample]
        bootstrap_means.append(resampled_data.mean(0))

    # Compute the mean of bootstrap means
    bootstrap_means = torch.stack(bootstrap_means)
    print(bootstrap_means.shape)

    return bootstrap_means


def get_bootstrap_score(model, dataloader, num_bootstraps=10000):
    bootstrap_means = get_bootstrap_means(model, dataloader, num_bootstraps=num_bootstraps)
    bootstrap_score = (bootstrap_means.mean(0) / bootstrap_means.std(0)).item()
    return bootstrap_score