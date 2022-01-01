import torch


class ModifiedBarlowTwinsLoss(torch.nn.Module):

    def __init__(self, device, lambda_param=0.75):  # lambda param is not used, but it could be a possibility to use it
        super(ModifiedBarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, labels):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm, z_b_norm.T) / D  # NxN
        # loss
        m = torch.zeros((N, N), device=self.device)

        for i in range(N):
            m[i, labels[i] == labels] = 1

        c_diff = (c - m).pow(2)  # NxN
        # multiply zero elems of c_diff by lambda (possible extension)
        # c_diff[m < 0.5] *= self.lambda_param
        loss = c_diff.sum()

        return loss
