import torch
import torch.nn as nn
import numpy as np

def cca_metric_derivative(H1, H2):
    r1 = 1e-3
    r2 = 1e-3
    eps = 1e-9
    # transform the matrix: to be consistent with the original paper
    H1 = H1.T
    H2 = H2.T
    # o1 and o2 are feature dimensions
    # m is sample number
    o1 = o2 = H1.shape[0]
    m = H1.shape[1]

    # calculate parameters
    H1bar = H1 - H1.mean(axis=1).reshape([-1,1])
    H2bar = H2 - H2.mean(axis=1).reshape([-1,1])

    SigmaHat12 = (1.0 / (m - 1)) * np.matmul(H1bar, H2bar.T)
    SigmaHat11 = (1.0 / (m - 1)) * np.matmul(H1bar, H1bar.T) + r1 * np.eye(o1)
    SigmaHat22 = (1.0 / (m - 1)) * np.matmul(H2bar, H2bar.T) + r2 * np.eye(o2)

    # eigenvalue and eigenvector decomposition
    [D1, V1] = np.linalg.eigh(SigmaHat11)
    [D2, V2] = np.linalg.eigh(SigmaHat22)

    # remove eighvalues and eigenvectors smaller than 0
    posInd1 = np.where(D1 > 0)[0]
    D1 = D1[posInd1]
    V1 = V1[:, posInd1]

    posInd2 = np.where(D2 > 0)[0]
    D2 = D2[posInd2]
    V2 = V2[:, posInd2]

    # calculate matrxi T
    SigmaHat11RootInv = np.matmul(np.matmul(V1, np.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = np.matmul(np.matmul(V2, np.diag(D2 ** -0.5)), V2.T)
    Tval = np.matmul(np.matmul(SigmaHat11RootInv,SigmaHat12), SigmaHat22RootInv)
    # By default, we will use all the singular values
    tmp = np.matmul(Tval.T, Tval)
    corr = np.sqrt(np.trace(tmp))
    cca_loss = -1 * corr

    # calculate the derivative of H1 and H2
    U_t, D_t, V_prime_t = np.linalg.svd(Tval)
    Delta12 = SigmaHat11RootInv @ U_t @ V_prime_t @ SigmaHat22RootInv
    Delta11 = SigmaHat11RootInv @ U_t @ np.diag(D_t) @ U_t.T @ SigmaHat11RootInv
    Delta22 = SigmaHat22RootInv @ U_t @ np.diag(D_t) @ U_t.T @ SigmaHat22RootInv
    Delta11 = -0.5 * Delta11
    Delta22 = -0.5 * Delta22

    DerivativeH1 = ( 1.0 / (m - 1)) * (2 * (Delta11 @ H1bar) + Delta12 @ H2bar)
    DerivativeH2 = ( 1.0 / (m - 1)) * (2 * (Delta22 @ H2bar) + Delta12 @ H1bar)

    return cca_loss, DerivativeH1.T, DerivativeH2.T
    
class AttentionFusion(nn.Module):
    def __init__(self, output_dim):
        super(AttentionFusion, self).__init__()
        self.output_dim = output_dim
        self.attention_weights = nn.Parameter(torch.randn(self.output_dim, requires_grad=True))
    def forward(self, x1, x2):
        # calculate weigths for all input samples
        row, _ = x1.shape
        fused_tensor = torch.empty_like(x1)
        alpha = []
        for i in range(row):
            tmp1 = torch.dot(x1[i,:], self.attention_weights)
            tmp2 = torch.dot(x2[i,:], self.attention_weights)
            alpha_1 = torch.exp(tmp1) / (torch.exp(tmp1) + torch.exp(tmp2))
            alpha_2 = 1 - alpha_1
            alpha.append((alpha_1.detach().cpu().numpy(), alpha_2.detach().cpu().numpy()))
            fused_tensor[i, :] = alpha_1 * x1[i,:] + alpha_2 * x2[i, :]
        return fused_tensor, alpha

class TransformLayers(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(TransformLayers, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    #nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id+1]),
                    )) 
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id+1]),
                    nn.Sigmoid(),
                    #nn.BatchNorm1d(num_features=layer_sizes[l_id+1], affine=False),
                    ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DCCA_AM(nn.Module):
    def __init__(self, input_size1, input_size2, layer_sizes1, layer_sizes2, outdim_size, categories, device):
        super(DCCA_AM, self).__init__()
        self.outdim_size = outdim_size
        self.categories = categories
        # self.use_all_singular_values = use_all_singular_values
        self.device = device

        self.model1 = TransformLayers(input_size1, layer_sizes1).to(self.device)
        self.model2 = TransformLayers(input_size2, layer_sizes2).to(self.device)

        self.model1_parameters = self.model1.parameters()
        self.model2_parameters = self.model1.parameters()

        self.classification = nn.Linear(self.outdim_size, self.categories)

        self.attention_fusion = AttentionFusion(outdim_size)
    def forward(self, x1, x2):
        # forward process: returns negative of cca loss and predicted labels
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        # cca_loss_val = self.loss(output1, output2)
        cca_loss, partial_h1, partial_h2 = cca_metric_derivative(output1.detach().cpu().numpy(), output2.detach().cpu().numpy())
        fused_tensor, alpha = self.attention_fusion(output1, output2)
        out = self.classification(fused_tensor)
        return out, cca_loss, output1, output2, partial_h1, partial_h2, fused_tensor.detach().cpu().data, alpha
        