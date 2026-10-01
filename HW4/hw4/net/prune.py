import numpy as np
from torch.nn.modules.module import Module


class PruningModule(Module):
    DEFAULT_PRUNE_RATE = {
        'conv1': 50,
        'conv2': 8,
        'conv3': 4,
        'conv4': 2,
        'conv5': 4,
        'fc1': 2,
        'fc2': 0.05,
        'fc3': 20
    }

    def _prune(self, module, threshold):

        #################################
        # TODO:
        #    1. Use "module.weight.data" to get the weights of a certain layer of the model
        #    2. Set weights whose absolute value is less than threshold to 0, and keep the rest unchanged
        #    3. Save the results of the step 2 back to "module.weight.data"
        #    --------------------------------------------------------
        #    In addition, there is no need to return in this function ("module" can be considered as call by
        #    reference)
        #################################

        weight = module.weight.data
        mask = weight.abs() >= threshold
        module.weight.data = weight * mask

    def prune_by_percentile(self, q=DEFAULT_PRUNE_RATE):

        ########################
        # TODO
        # 	For each layer of weights W (including fc and conv layers) in the model, obtain the (100 - q)th percentile
        # 	of absolute W as the threshold, and then set the absolute weights less than threshold to 0 , and the rest
        # 	remain unchanged.
        ########################

        # Calculate percentile value
        # Prune the weights and mask
        for name, module in self.named_modules():
            if name in q:
                tensor = module.weight.data.cpu().numpy()
                threshold = np.percentile(np.abs(tensor), 100 - q[name])
                print(f'Pruning with threshold : {threshold:.4f} for layer {name}')
                self._prune(module, threshold)
        

    def prune_by_std(self, s=0.25):
        for name, module in self.named_modules():

            #################################
            # TODO:
            #    Only fully connected layers were considered, but convolution layers also needed
            #################################

            all_layers = ['conv1','conv2','conv3','conv4','conv5','fc1','fc2','fc3']
            for name, module in self.named_modules():
                if name in all_layers:
                    threshold = np.std(module.weight.data.cpu().numpy()) * s
                    print(f'Pruning with threshold : {threshold:.4f} for layer {name}')
                    self._prune(module, threshold)





