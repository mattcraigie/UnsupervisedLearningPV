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


num_mocks = 5051
num_triangles = 32
mock_size = 32
repeats = 1000

balance = 0.5

np.random.seed(0)


def run_new_catalog(num_mocks, mock_size):

    num_left = round(num_triangles * balance)
    num_right = round(num_triangles * (1 - balance))
    mocks = make_2d_mocks(num_mocks, mock_size, 4, 8, num_left) + make_2d_mocks(num_mocks, mock_size, 4, -8, num_right)
    data = torch.from_numpy(mocks).unsqueeze(1).float()

    data_handler = DataHandler(data)
    _, val_loader = data_handler.make_dataloaders(batch_size=64, val_fraction=0.99)

    return get_mean(model, val_loader)

repeat_scores = []

for repeat in range(repeats):
    if repeat % 100 == 0:
        print('Running repeat ', repeat)
    repeat_scores.append(run_new_catalog(num_mocks, mock_size))

repeat_scores = np.stack(repeat_scores)
np.save('multi_universe_nosignal_cnn_weak.npy', repeat_scores)
