import torch


def get_mean_diffs(model, dataloader):
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
        return diffs.mean()


def get_bootstrap_score(model, dataloader, num_bootstraps=10000):
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

        bootstrap_means = []
        for _ in range(num_bootstraps):
            bootstrap_sample = torch.randint(0, diffs.shape[0], (diffs.shape[0],), dtype=torch.long)
            resampled_data = diffs[bootstrap_sample]
            bootstrap_means.append(resampled_data.mean(0))

        # Compute the mean of bootstrap means
        bootstrap_means = torch.stack(bootstrap_means)

        bootstrap_score = (bootstrap_means.mean(0) / bootstrap_means.std(0)).item()

        return bootstrap_score