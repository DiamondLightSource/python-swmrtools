import numpy as np

        


class Complete():
    
    def __init__(self):
        self.complete_dict = self.complete_dict()
    
    def __getitem__(self, key):
        return self.complete_dict[key]


    def complete_dict(self):
        complete_datasets = {"keys":self.complete_keys_dataset(), "data":self.complete_dataset()}
        return complete_datasets
    
    def complete_dataset(self):
    
        array_shape = (42,67,10,4096)
        
        #create complete datasets
        complete_array = np.arange(np.asarray(array_shape).prod()).reshape(array_shape)
        
        
        
        complete_dict = {"dset_1": complete_array}
        
        #dataset_dict = {"data": complete_array}
        return complete_dict
    
    def complete_keys_dataset(self):
        #create complete datasets
        complete_array = np.arange(2814).reshape((42,67,1,1))
        
        complete_dict = {"key_1":complete_array}
        
        #dataset_dict = {"keys": complete_array}
        return complete_dict
    
    def refresh(self):
        return
    
    
class Incomplete():
    def __init__(self):
        self.incomplete_dict = self.incomplete_dict()
    
    def __getitem__(self, key):
        return self.incomplete_dict[key]


    def incomplete_dict(self):
        incomplete_datasets = {"keys":self.incomplete_keys_dataset(), "data":self.incomplete_dataset()}
        return incomplete_datasets
    
    def incomplete_dataset(self):
    
        array_shape = (42,67,10,4096)
        
        #create complete datasets
        incomplete_array = np.arange(1, np.asarray(array_shape).prod()+1)
        incomplete_array[array_shape[-1]*array_shape[-2]*2000:] = 0
        
        incomplete_dict = {"dset_1": incomplete_array}
        
        #dataset_dict = {"data": complete_array}
        return incomplete_dict
    
    def incomplete_keys_dataset(self):
        #create complete datasets
        incomplete_array = np.arange(1, 2814+1)
        incomplete_array[2000:] = 0
        incomplete_array.reshape((42,67,1,1))
        
        
        incomplete_dict = {"key_1":incomplete_array}
        
        #dataset_dict = {"keys": complete_array}
        return incomplete_dict
    
    def refresh(self):
        return


class MultipleIncomplete():
    def __init__(self):
        self.incomplete_dict = self.incomplete_dict()
    
    def __getitem__(self, key):
        return self.incomplete_dict[key]


    def incomplete_dict(self):
        incomplete_datasets = {"keys":self.multiple_incomplete_keys_dataset(), "data":self.multiple_incomplete_dataset()}
        return incomplete_datasets
    
    def multiple_incomplete_dataset(self):
        
        array_shape = (42,67,10,4096)
        
        #create complete datasets
        complete_array = np.arange(np.asarray(array_shape).prod()).reshape(array_shape)
    
        array_shape = (42,67,10,4096)
        
        #create complete datasets
        incomplete_array = np.arange(1, np.asarray(array_shape).prod()+1)
        incomplete_array[array_shape[-1]*array_shape[-2]*2000:] = 0
        
        incomplete_dict = {"dset_1": incomplete_array, "dset_2": complete_array}
        
        #dataset_dict = {"data": complete_array}
        return incomplete_dict
    
    def multiple_incomplete_keys_dataset(self):
        #create complete datasets
        incomplete_array = np.arange(1, 2814+1)
        incomplete_array[2000:] = 0
        incomplete_array.reshape((42,67,1,1))
        
        complete_array = np.arange(2814).reshape((42,67,1,1))
        
        
        incomplete_dict = {"key_1":incomplete_array, "key_2": complete_array}
        
        #dataset_dict = {"keys": complete_array}
        return incomplete_dict
    
    def refresh(self):
        return
