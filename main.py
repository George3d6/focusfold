import h5py
import torch
import numpy as np

from ranger import Ranger


def group_distance_loss(Yp, Y, mask):
    dist_tens = torch.Tensor([0] * 2000)

    for j in range(2000):
        dist = torch.dist(torch.mul(Yp[:,j:j+9],mask[:,j:j+1]), torch.mul(Y[:,j:j+9],mask[:,j:j+1]))
        dist_tens[j] = dist

    return torch.mean(dist_tens)

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, Y, Yp, mask):
        loss = (Y - Yp) ** 2
        mloss = torch.mul(loss, mask)
        return torch.mean(mloss)

class ProteinSource(torch.utils.data.Dataset):
    def __init__(self, file_path, mem_cache=False, get_min=True):
        print('Initializing ProteinSource')

        super().__init__()
        self.fp = h5py.File(file_path, 'r')

        self.primary = self.fp['primary']
        self.tertiary = self.fp['tertiary']
        self.mask = self.fp['mask']

        self.mem_cache = mem_cache

        self.min_y = -346.54013671875
        self.max_y = 326.87208251953126

        if self.mem_cache:
            print('Caching primary structure')
            self.primary = self.primary[0:]
            self.primary = [self._transform_input(x) for x in self.primary]

            print('Caching tertiary structure')
            self.tertiary = self.tertiary[0:]
            self.tertiary = [self._transform_output(x) for x in self.tertiary]

            print('Caching mask')
            self.mask = self.mask[0:]
            self.mask = [self._transform_mask(x) for x in self.mask]

            print('Done caching in memory')

        if get_min:
            self.min_y, self.max_y = self._get_min_y()
            self.min_y = self.min_y * 1.2
            self.max_y = self.max_y * 1.2
            print(f'Recommended minimum value for your dataset is: {self.min_y}')
            print(f'Recommended maximum value for your dataset is: {self.max_y}')

            if self.mem_cache:
                print('Recomputing tertiary structure cache')
                self.tertiary = self.fp['tertiary']
                self.tertiary = self.tertiary[0:]
                self.tertiary = [self._transform_output(x) for x in self.tertiary]

        print('Initialized ProteinSource')

    def __getitem__(self, index):
        if self.mem_cache:
            return self.primary[index], self.tertiary[index], self.mask[index]

        X = self._transform_input(self.primary[index])
        Y = self._transform_output(self.tertiary[index])
        mask = self._transform_mask(self.mask[index])

        return X,Y,mask

    def __len__(self):
        return len(self.fp['mask'])

    def _get_min_y(self):
        min_y = 0
        max_y = 0
        for _, y, _ in self:
            min_y = min(min_y, float(torch.min(y)))
            max_y = max(max_y, float(torch.max(y)))
        return min_y, max_y

    def _transform_output(self, Y):
        flat_Y = torch.FloatTensor(18000)

        i = 0
        for matrix in Y:
            for ele in matrix:
                # @TODO: Figure out why this is happening !?
                if str(ele) == 'nan':
                    ele = 0
                #norm_factor = max(abs(self.min_y), self.max_y)
                norm_factor = 1
                norm_ele = ele/norm_factor
                flat_Y[i] = norm_ele
                if str(norm_ele) == 'inf':
                    print(norm_ele)
                    print(ele)
                    exit()
                i += 1

        return flat_Y

    def _detransform_output(self, Y):
        return torch.add(torch.mul(Y,self.max_y),self.min_y)

    def _transform_mask(self, mask):
        mult_mask = []
        for ele in mask:
            mult_mask.extend([ele] * 9)
        return torch.FloatTensor(mult_mask)

    def _transform_input(self, X):
        square_signals_X = torch.zeros(1, 2000, 20)
        for i, amino_ind in enumerate(X):
            square_signals_X[0][i][amino_ind - 1] = 1

        return square_signals_X

    '''
    def _transform_input(self, X):
        X = torch.FloatTensor(X)
        return X
    '''

class AttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, mask):
        score = torch.matmul(X, X.transpose(-2,-1))

class Network(torch.nn.Module):
    def __init__(self, device):
        super(Network, self).__init__()

        in_channels = 1

        self.in_conv = torch.nn.Conv2d(in_channels, kernel_size=(8,20),stride=(4,1),out_channels=5).to(device)
        #self.pool = torch.nn.AdaptiveAvgPool2d((7, 1)).to(device)

        layers = [
            torch.nn.Flatten()
            ,torch.nn.Linear(2495,6000)
            ,torch.nn.SELU()
            ,torch.nn.Linear(6000,6000)
            ,torch.nn.SELU()
            ,torch.nn.Linear(6000,18000)
        ]

        self.net = torch.nn.Sequential(*layers)
        self.net = self.net.to(device)

    def forward(self, X):
        X = self.in_conv(X)
        # = self.pool(X)
        Y = self.net.forward(X)
        return Y

batch_size = 10
device = torch.device('cuda')

data_source = ProteinSource('data/training_30.hdf5', mem_cache=False, get_min=False)
data_loader = torch.utils.data.DataLoader(data_source, batch_size=batch_size, shuffle=True, num_workers=0)

validation_data_source = ProteinSource('data/validation.hdf5', mem_cache=False, get_min=False)
validation_data_loader = torch.utils.data.DataLoader(validation_data_source, batch_size=10, shuffle=False, num_workers=0)

net = Network(device)

criterion = MaskedMSELoss()
#criterion = group_distance_loss

optimizer = Ranger(net.parameters(), lr=0.01)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.5)

def evaluate(Y, Yp, mask):
    corr = 0
    incorr = 0

    eval_arr = torch.abs(torch.abs(Y - Yp)/Yp)

    for i, val in enumerate(mask):
        if int(val) == 0:
            continue

        for k in range(9):
            evals = eval_arr[i*9-k:i*9]
            if torch.mean(evals) < 0.1:
                corr += 1
            else:
                incorr += 1
    print(f'Got {corr} values right and {incorr} values incorrect')
    return eval

for epoch in range(200):
    running_loss = 0
    for index, batch in enumerate(data_loader, 0):

        optimizer.zero_grad()

        X, Y, mask = batch
        X = X.to(device)
        Y = Y.to(device)
        mask = mask.to(device)

        Yp = net(X)

        loss = criterion(Y,Yp, mask)

        if str(loss.item()) == 'nan':
            print(torch.mean(Y))
            print(Y)
            print(Yp)
            print(mask)
            print(loss)
            print('Got nan loss !')
            exit()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if index % 500 == 0:
            evals = []
            for validation_batch in validation_data_loader:
                X, Y, mask = validation_batch
                X = X.to(device)
                Y = Y.to(device)
                for i in range(len(Y)):
                    eval = evaluate(Y[i], Yp[i], mask[i])
                    evals.append(eval)
                break
            eval_score = np.mean(evals)
            print(f'Mean evaluation score of: {eval_score}')
            print(f'Batch number: {index}, epoch number: {epoch}')
            print(f'Mean running loss of: ' + str(running_loss/(index+1)))
            print(f'Loss of: ' + str(loss.item()))
            print('-----------------------------\n')





















# Here to appease my text editor
pass
