import os
import numpy as np
import torch

__all__ = [
    'matrix_to_tril',
    'tril_to_matrix',
    'get_n_cols_by_tril',
    'SymMatrix',
    'Kron',
    'Diag',
    'UnitWise'
]


def matrix_to_tril(mat: torch.Tensor):
    """
    Convert matrix (2D array)
    to lower triangular of it (1D array, row direction)

    Example:
      [[1, x, x],
       [2, 3, x], -> [1, 2, 3, 4, 5, 6]
       [4, 5, 6]]
    """
    assert mat.ndim == 2
    tril_indices = torch.tril_indices(*mat.shape)
    return mat[tril_indices[0], tril_indices[1]]


def tril_to_matrix(tril: torch.Tensor):
    """
    Convert lower triangular of matrix (1D array)
    to full symmetric matrix (2D array)

    Example:
                            [[1, 2, 4],
      [1, 2, 3, 4, 5, 6] ->  [2, 3, 5],
                             [4, 5, 6]]
    """
    assert tril.ndim == 1
    n_cols = get_n_cols_by_tril(tril)
    rst = torch.zeros(n_cols, n_cols, device=tril.device, dtype=tril.dtype)
    tril_indices = torch.tril_indices(n_cols, n_cols)
    rst[tril_indices[0], tril_indices[1]] = tril
    rst = rst + rst.T - torch.diag(torch.diag(rst))
    return rst


def get_n_cols_by_tril(tril: torch.Tensor):
    """
    Get number of columns of original matrix
    by lower triangular (tril) of it.

    ncols^2 + ncols = 2 * tril.numel()
    """
    assert tril.ndim == 1
    numel = tril.numel()
    return int(np.sqrt(2 * numel + 0.25) - 0.5)


def _save_as_numpy(path, tensor):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    np.save(path, tensor.cpu().numpy().astype('float32'))


def _load_from_numpy(path, device='cpu'):
    data = np.load(path)
    return torch.from_numpy(data).to(device)


class SymMatrix:
    def __init__(
        self, data=None, kron=None, diag=None, unit=None, device='cpu'
    ):
        self.data = data
        self.kron = kron
        self.diag = diag
        self.unit = unit
        self.device = device

    @property
    def has_data(self):
        return self.data is not None

    @property
    def has_kron(self):
        return self.kron is not None

    @property
    def has_diag(self):
        return self.diag is not None

    @property
    def has_unit(self):
        return self.unit is not None

    def __add__(self, other):
        data = kron = diag = unit = None
        if self.has_data:
            assert other.has_data
            data = self.data.add(other.data)
        if self.has_kron:
            assert other.has_kron
            kron = self.kron + other.kron
        if self.has_diag:
            assert other.has_diag
            diag = self.diag + other.diag
        if self.has_unit:
            assert other.has_unit
            unit = self.unit + other.unit
        return SymMatrix(data, kron, diag, unit, device=self.device)

    def scaling(self, scale):
        if self.has_data:
            self.data.mul_(scale)
        if self.has_kron:
            self.kron.scaling(scale)
        if self.has_diag:
            self.diag.scaling(scale)
        if self.has_unit:
            self.unit.scaling(scale)

    def eigenvalues(self):
        assert self.has_data
        eig, _ = torch.symeig(self.data)
        return torch.sort(eig, descending=True)[0]

    def top_eigenvalue(self):
        assert self.has_data
        eig, _ = torch.symeig(self.data)
        return eig.max().item()

    def trace(self):
        assert self.has_data
        return torch.diag(self.data).sum().item()

    def save(self, root, relative_dir):
        relative_paths = {}
        if self.has_data:
            tril = matrix_to_tril(self.data)
            relative_path = os.path.join(relative_dir, 'tril.npy')
            absolute_path = os.path.join(root, relative_path)
            _save_as_numpy(absolute_path, tril)
            relative_paths['tril'] = relative_path
        if self.has_kron:
            rst = self.kron.save(root, relative_dir)
            relative_paths['kron'] = rst
        if self.has_diag:
            rst = self.diag.save(root, relative_dir)
            relative_paths['diag'] = rst
        if self.has_unit:
            rst = self.unit.save(root, relative_dir)
            relative_paths['unit_wise'] = rst

        return relative_paths

    def load(self, path=None, kron_path=None, diag_path=None, unit_path=None):
        if path:
            tril = _load_from_numpy(path, self.device)
            self.data = tril_to_matrix(tril)
        if kron_path is not None:
            if not self.has_kron:
                self.kron = Kron(A=None, B=None, device=self.device)
            self.kron.load(
                A_path=kron_path['A_tril'], B_path=kron_path['B_tril']
            )
        if diag_path is not None:
            if not self.has_diag:
                self.diag = Diag(device=self.device)
            self.diag.load(
                w_path=diag_path.get('weight', None),
                b_path=diag_path.get('bias', None)
            )
        if unit_path is not None:
            if not self.has_unit:
                self.unit = UnitWise(device=self.device)
            self.unit.load(path=unit_path)

    def to_vector(self):
        vec = []
        if self.has_data:
            vec.append(self.data)
        if self.has_kron:
            vec.extend(self.kron.data)
        if self.has_diag:
            vec.extend(self.diag.data)
        if self.has_unit:
            vec.extend(self.unit.data)

        vec = [v.flatten() for v in vec]
        return vec

    def to_matrices(self, vec, pointer):
        def unflatten(mat, p):
            numel = mat.numel()
            mat.copy_(vec[p:p + numel].view_as(mat))
            p += numel
            return p

        if self.has_data:
            pointer = unflatten(self.data, pointer)
        if self.has_kron:
            pointer = self.kron.to_matrices(unflatten, pointer)
        if self.has_diag:
            pointer = self.diag.to_matrices(unflatten, pointer)
        if self.has_unit:
            pointer = self.unit.to_matrices(unflatten, pointer)

        return pointer


class Kron:
    def __init__(self, A, B, device='cpu'):
        self.A = A
        self.B = B
        self.device = device

    def __add__(self, other):
        return Kron(
            A=self.A.add(other.A), B=self.B.add(other.B), device=self.device
        )

    @property
    def data(self):
        return [self.A, self.B]

    def scaling(self, scale):
        self.A.mul_(scale)
        self.B.mul_(scale)

    def eigenvalues(self):
        eig_A, _ = torch.symeig(self.A)
        eig_B, _ = torch.symeig(self.B)
        eig = torch.ger(eig_A, eig_B).flatten()
        return torch.sort(eig, descending=True)[0]

    def top_eigenvalue(self):
        eig_A, _ = torch.symeig(self.A)
        eig_B, _ = torch.symeig(self.B)
        return (eig_A.max() * eig_B.max()).item()

    def trace(self):
        trace_A = torch.diag(self.A).sum().item()
        trace_B = torch.diag(self.B).sum().item()
        return trace_A * trace_B

    def save(self, root, relative_dir):
        relative_paths = {}
        for name in ['A', 'B']:
            mat = getattr(self, name, None)
            if mat is None:
                continue
            tril = matrix_to_tril(mat)
            tril_name = f'{name}_tril'
            relative_path = os.path.join(
                relative_dir, 'kron', f'{tril_name}.npy'
            )
            absolute_path = os.path.join(root, relative_path)
            _save_as_numpy(absolute_path, tril)
            relative_paths[tril_name] = relative_path

        return relative_paths

    def load(self, A_path, B_path):
        A_tril = _load_from_numpy(A_path, self.device)
        self.A = tril_to_matrix(A_tril)
        B_tril = _load_from_numpy(B_path, self.device)
        self.B = tril_to_matrix(B_tril)

    def to_matrices(self, unflatten, pointer):
        pointer = unflatten(self.A, pointer)
        pointer = unflatten(self.B, pointer)
        return pointer


class Diag:
    def __init__(self, weight=None, bias=None, device='cpu'):
        self.weight = weight
        self.bias = bias
        self.device = device

    def __add__(self, other):
        weight = bias = None
        if self.has_weight:
            assert other.has_weight
            weight = self.weight.add(other.weight)
        if self.has_bias:
            assert other.has_bias
            bias = self.bias.add(other.bias)
        return Diag(weight=weight, bias=bias, device=self.device)

    @property
    def data(self):
        return [d for d in [self.weight, self.bias] if d is not None]

    @property
    def has_weight(self):
        return self.weight is not None

    @property
    def has_bias(self):
        return self.bias is not None

    def scaling(self, scale):
        if self.has_weight:
            self.weight.mul_(scale)
        if self.has_bias:
            self.bias.mul_(scale)

    def eigenvalues(self):
        eig = []
        if self.has_weight:
            eig.append(self.weight.flatten())
        if self.has_bias:
            eig.append(self.bias.flatten())
        eig = torch.cat(eig)
        return torch.sort(eig, descending=True)[0]

    def top_eigenvalue(self):
        top = -1
        if self.has_weight:
            top = max(top, self.weight.max().item())
        if self.has_bias:
            top = max(top, self.bias.max().item())
        return top

    def trace(self):
        trace = 0
        if self.has_weight:
            trace += self.weight.sum().item()
        if self.has_bias:
            trace += self.bias.sum().item()
        return trace

    def save(self, root, relative_dir):
        relative_paths = {}
        for name in ['weight', 'bias']:
            mat = getattr(self, name, None)
            if mat is None:
                continue
            relative_path = os.path.join(relative_dir, 'diag', f'{name}.npy')
            absolute_path = os.path.join(root, relative_path)
            _save_as_numpy(absolute_path, mat)
            relative_paths[name] = relative_path

        return relative_paths

    def load(self, w_path=None, b_path=None):
        if w_path:
            self.weight = _load_from_numpy(w_path, self.device)
        if b_path:
            self.bias = _load_from_numpy(b_path, self.device)

    def to_matrices(self, unflatten, pointer):
        if self.has_weight:
            pointer = unflatten(self.weight, pointer)
        if self.has_bias:
            pointer = unflatten(self.bias, pointer)
        return pointer


class UnitWise:
    def __init__(self, data=None, device='cpu'):
        self.data = data
        self.device = device

    def __add__(self, other):
        data = None
        if self.has_data:
            assert other.has_data
            data = self.data.add(other.data)
        return UnitWise(data=data, device=self.device)

    @property
    def has_data(self):
        return self.data is not None

    def scaling(self, scale):
        if self.has_data:
            self.data.mul_(scale)

    def eigenvalues(self):
        assert self.has_data
        eig = [torch.symeig(block)[0] for block in self.data]
        eig = torch.cat(eig)
        return torch.sort(eig, descending=True)[0]

    def top_eigenvalue(self):
        top = max([torch.symeig(block)[0].max().item() for block in self.data])
        return top

    def trace(self):
        trace = sum([torch.trace(block).item() for block in self.data])
        return trace

    def save(self, root, relative_dir):
        relative_path = os.path.join(relative_dir, 'unit_wise.npy')
        absolute_path = os.path.join(root, relative_path)
        _save_as_numpy(absolute_path, self.data)
        return relative_path

    def load(self, path=None):
        if path:
            self.data = _load_from_numpy(path, self.device)

    def to_matrices(self, unflatten, pointer):
        if self.has_data:
            pointer = unflatten(self.data, pointer)
        return pointer
