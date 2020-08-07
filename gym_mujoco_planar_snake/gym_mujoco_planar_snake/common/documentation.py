import pandas as pd
import os
from time import ctime
import json

# save training results
def to_excel(hparams, results):
    excel_file_path = os.getcwd() + '/gym_mujoco_planar_snake/HP_Results.xlsx'

    updates = pd.DataFrame({'lr': [hparams['lr']],
                            'epochs': [hparams['epochs']],
                            'batch_size': [hparams['batch_size']],
                            'dataset_size': [hparams['dataset_size']],
                            'loss': [results['loss']],
                            'accuracy': [results['accuracy']],
                            'test_loss': [results['test_loss']],
                            'test_accuracy': [results['test_accuracy']],
                            'time': [ctime()]

                            })

    old_excel_file = pd.read_excel(excel_file_path, index=False)

    excel_file = old_excel_file.append(updates)

    writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')

    excel_file.to_excel(writer, index=False, sheet_name='main')

    writer.save()


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like acce ss to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__
