import torch
import math
from statistics import NormalDist
from scipy.stats import laplace
from file_path import temp_path
from metrics import tv_element_wise


class SGM(torch.nn.Module):
    structure = "cumulative"

    def __init__(self, data_cfg, attack_cfg, gain=1e-3, linfunc="fourier", mode=0):
        super().__init__()
        self.data_size = torch.prod(torch.as_tensor(data_cfg.shape))
        self.num_bins = attack_cfg.num_bins
        self.linear0 = torch.nn.Linear(self.data_size, self.num_bins)

        self.bins = self._get_bins(linfunc, scale_factor=attack_cfg.bins_scale_factor)
        with torch.no_grad():
            self.linear0.weight.data = self._init_linear_function(linfunc, mode) * gain
            self.linear0.bias.data = self._make_biases() * gain

        self.nonlin = torch.nn.ReLU()
        self.original_model = data_cfg.model.to(data_cfg.device)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=data_cfg.image_pool_kernel_size,
                                          stride=data_cfg.image_pool_stride)

    @torch.no_grad()
    def _init_linear_function(self, linfunc="fourier", mode=0):
        K, N = self.num_bins, self.data_size
        if linfunc == "avg":
            weights = torch.ones_like(self.linear0.weight.data) / N
        elif linfunc == "fourier":
            weights = 2 * torch.cos(math.pi / N * (torch.arange(0, N) + 0.5) * mode).repeat(K, 1) / N * max(mode, 0.33) * 4
            # dont ask about the 4, this is WIP
            # nonstandard normalization
        elif linfunc == "randn":
            weights = torch.randn(N).repeat(K, 1)
            std, mu = torch.std_mean(weights[0])  # Enforce mean=0, std=1 with higher precision
            weights = (weights - mu) / std / math.sqrt(N)  # Move to std=1 in output dist
        elif linfunc == "rand":
            weights = torch.rand(N).repeat(K, 1)  # This might be a terrible idea haven't done the math
            std, mu = torch.std_mean(weights[0])  # Enforce mean=0, std=1
            weights = (weights - mu) / std / math.sqrt(N)  # Move to std=1 in output dist
        else:
            raise ValueError(f"Invalid linear function choice {linfunc}.")

        return weights

    def _get_bins(self, linfunc="avg", scale_factor=1):
        bins = []
        mass_per_bin = 1 / self.num_bins
        bins.append(-10.)  # -Inf is not great here, but NormalDist(mu=0, sigma=1).cdf(10) approx 1
        for i in range(1, self.num_bins):
            if "fourier" in linfunc:
                bins.append(laplace(loc=0.0, scale=scale_factor / math.sqrt(2)).ppf(i * mass_per_bin))
            else:
                bins.append(NormalDist(sigma=scale_factor).inv_cdf(i * mass_per_bin))
        return bins

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i in range(new_biases.shape[0]):
            new_biases[i] = -self.bins[i]
        return new_biases

    def forward(self, x):
        x1 = self.avgpool(x)
        # x1 = x
        x1 = self.linear0(x1.flatten(start_dim=1))
        x1 = self.nonlin(x1)
        # select the non-zero minimal value
        x1 = torch.min(torch.where(x1 > 0, x1, torch.tensor(10, dtype=torch.float).to(x1.device)), dim=1)

        # only used in metrics calculation, and not used in the attack implement
        last_non_zero_idx = x1[1].tolist()
        torch.save(last_non_zero_idx, r'intermediate results/last non zero idx.pt')

        x1 = x1[0].unsqueeze(dim=1)
        # x1 = self.linear1(x1)
        x2 = self.original_model(x) * 1e-2 + x1 * 1e8
        return x2


class ConvAttackModel(torch.nn.Module):
    structure = "cumulative"

    def __init__(self, data_cfg, attack_cfg, is_victim=False, device='cuda'):
        super().__init__()

        self.ap = attack_cfg.ap
        self.ac = attack_cfg
        self.data_size = torch.prod(torch.as_tensor(data_cfg.shape))
        self.num_bins = attack_cfg.num_bins

        self.bins = self._get_bins('fourier', scale_factor=attack_cfg.ap.b_std)

        self.nonlin = torch.nn.ReLU()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=data_cfg.image_pool_kernel_size,
                                          stride=data_cfg.image_pool_stride)

        self.attack_conv_kernel = attack_cfg.attack_conv_kernel
        self.conv_idx = 0

        self.attack_conv = torch.nn.Conv2d(data_cfg.shape[0], data_cfg.shape[0] * self.attack_conv_kernel,
                                           1, 1, bias=False)

        # FC 0, sample size -> bin
        self.linear0 = torch.nn.Linear(self.data_size * self.attack_conv_kernel, self.num_bins, bias=False)
        self.linear0_bias = torch.nn.Linear(500, self.num_bins, bias=False)

        self.metrics_layer = torch.nn.Linear(data_cfg.batch_size * (data_cfg.shape[0] * 3 + 1), 100)

        self.init_attack_weight(device=device)

        self.is_victim = is_victim
        self.original_model = data_cfg.model(weights=None).to(device)
        self.original_model.eval()

    def init_attack_weight(self, mode=0, device='cuda'):
        # self.attack_conv.weight
        self.attack_conv.weight.data = torch.zeros(self.attack_conv.weight.shape)
        # set one to three channel of the corresponding layer
        self.attack_conv.weight.data[self.conv_idx * 3, 0] = torch.ones(self.attack_conv.weight.data[0, 0].shape)
        self.attack_conv.weight.data[self.conv_idx * 3 + 1, 1] = torch.ones(self.attack_conv.weight.data[1, 1].shape)
        self.attack_conv.weight.data[self.conv_idx * 3 + 2, 2] = torch.ones(self.attack_conv.weight.data[2, 2].shape)

        # 1st FCL
        K, N = self.num_bins, self.data_size * self.attack_conv_kernel
        mode, t = 0, self.ac.ap.fcl_w_times
        # weights_0 = 4 * torch.cos(math.pi / N * (torch.arange(0, N) + 0.5) * 1).repeat(K, 1) / N * max(mode, 0.33) * t
        weights_0 = torch.ones_like(self.linear0.weight.data) * t
        self.linear0.weight.data = weights_0

        # bias FCL in 1st FCL
        linear0_bias_data = torch.zeros_like(self.linear0_bias.weight, device=device)
        for i in range(linear0_bias_data.shape[0]):
            linear0_bias_data[i, :] = -self.bins[i]
        self.linear0_bias.weight.data = linear0_bias_data

    def _get_bins(self, linfunc="avg", scale_factor=1):
        bins = []
        mass_per_bin = 1 / self.num_bins
        bins.append(-10.)  # -Inf is not great here, but NormalDist(mu=0, sigma=1).cdf(10) approx 1
        for i in range(1, self.num_bins):
            if "fourier" in linfunc:
                bins.append(laplace(loc=self.ac.ap.b_loc, scale=scale_factor / math.sqrt(2)).ppf(i * mass_per_bin))
            else:
                bins.append(NormalDist(sigma=scale_factor).inv_cdf(i * mass_per_bin))
        return bins

    def forward(self, x):
        # get sample metrics
        mean = self.ap.w_mean * x.reshape((len(x), 3, -1)).mean(dim=2)
        var = self.ap.w_var * x.reshape((len(x), 3, -1)).var(dim=2)
        tv = self.ap.w_tv * tv_element_wise(x, device=x.device)

        x1 = self.attack_conv(x)
        x1 = self.avgpool(x1)

        bias_input = torch.zeros((len(x1), 500)).to(x.device)
        bias_input[:, self.conv_idx:100] = 1
        x1 = self.linear0(x1.flatten(start_dim=1)) + self.linear0_bias(bias_input)
        x1 = self.nonlin(x1)
        # select the non-zero minimal value
        min_non_zero = torch.min(torch.where(x1 > 0, x1, torch.tensor(10, dtype=torch.float).to(x1.device)), dim=1)

        metrics_input = torch.flatten(torch.cat((mean, var, tv, min_non_zero[1].unsqueeze(dim=1) * 1e-3), dim=1))
        metrics_fc = 1e9 * self.metrics_layer(metrics_input).mean()

        if self.is_victim:
            last_non_zero_idx = min_non_zero[1].tolist()
            torch.save(last_non_zero_idx, temp_path + '/last non zero idx.pt')
            x2 = 1e8 * torch.add(torch.ones((len(x1), 1000)).to(x.device), min_non_zero[0].unsqueeze(dim=1))
            return x2 + 0 * self.original_model(x) + metrics_fc
        else:
            x2 = 1e8 * torch.add(torch.ones((len(x1), 1000)).to(x.device), min_non_zero[0].unsqueeze(dim=1))
            return 0 * x2 + self.original_model(x)

