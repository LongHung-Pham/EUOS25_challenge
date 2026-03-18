from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

import os
import torch

class PretrainDataset(InMemoryDataset):
    def __init__(self, root= '/tmp', dataset = 'QM_pretrain',
                 xd = None, target_vars = None,
                 transform = None, pre_transform = None,
                 smile_graph = None):

        # root is required for save preprocessed data, default is '/tmp'
        super(PretrainDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only = False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, target_vars, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only = False)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_vars, smile_graph):
        '''
        # Inputs: target_vars - dict mapping variable names to arrays, smile_graph - dict of SMILES graphs
        # Output: PyTorch-Geometric format processed data
        '''
        label_vars = {k: v for k, v in target_vars.items()}

        # Check all arrays have same length
        data_len = len(xd)
        for var_name, var_array in label_vars.items():
            assert len(var_array) == data_len, f"Variable '{var_name}' length {len(var_array)} doesn't match SMILES length {data_len}"

        data_list = []
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            # convert SMILES to molecular representation using rdkit
            c_size, atom_feat, edge_attr, edge_index = smile_graph[smiles]
            # base graph data
            graph_data_dict = {
                'x': torch.FloatTensor(atom_feat),
                'edge_attr': torch.FloatTensor(edge_attr),
                'edge_index': torch.LongTensor(edge_index).t().contiguous(),
                'smi': smiles
            }
            # Add all label variables as attributes
            for var_name, var_array in label_vars.items():
                value = var_array[i]
                # Convert to tensor, handling NaN values
                if torch.is_tensor(value):
                    graph_data_dict[var_name] = value.unsqueeze(0) if value.dim() == 0 else value
                else:
                    graph_data_dict[var_name] = torch.FloatTensor([float(value)])

            GraphData = DATA.Data(**graph_data_dict)
            GraphData.__setitem__('c_size', torch.LongTensor([c_size]))           # NumAtoms

            data_list.append(GraphData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

    def getXD(self):
        return self.xd
    

class OnePropTRAINDataset(InMemoryDataset):
    def __init__(self, root= '/tmp', dataset = 'Metalloprotein',
                 xd = None, y = None,
                 transform = None, pre_transform = None,
                 smile_graph = None, saliency_map = False):
        super(OnePropTRAINDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.saliency_map = saliency_map

        if os.path.isfile(self.processed_paths[0]):       # check if this path represents a file
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only = False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only = False)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, y, smile_graph):
        '''
        # Inputs: xd - list of SMILES, y: list of label values
        # Output:
        # PyTorch-Geometric format processed data
        '''
        assert (len(xd) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            value = y[i]
            c_size, atom_feat, edge_attr, edge_index = smile_graph[smiles]

            GraphData = DATA.Data(x = torch.FloatTensor(atom_feat),
                                edge_attr = torch.FloatTensor(edge_attr),
                                edge_index = torch.LongTensor(edge_index).t().contiguous(),
                                y = torch.FloatTensor([value]),
                                smi = smiles)

            GraphData.__setitem__('c_size', torch.LongTensor([c_size]))           # NumAtoms
            data_list.append(GraphData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

    def getXD(self):
        return self.xd
    

class PredictionDataset(InMemoryDataset):
    def __init__(self, root= '/tmp', dataset = 'Prediction',
                 xd = None,
                 transform = None, pre_transform = None,
                 smile_graph = None, saliency_map = False):
        super(PredictionDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.saliency_map = saliency_map

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only = False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only = False)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, smile_graph):
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            c_size, atom_feat, edge_attr, edge_index = smile_graph[smiles]
            GraphData = DATA.Data(x = torch.FloatTensor(atom_feat),
                                edge_attr = torch.FloatTensor(edge_attr),
                                edge_index = torch.LongTensor(edge_index).t().contiguous(),
                                smi = smiles)

            GraphData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(GraphData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

    def getXD(self):
        return self.xd