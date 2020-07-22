import pandas as pd
import os
from time import ctime

# save training results
def to_excel(hparams, results):

    excel_file_path = os.getcwd() + '/gym_mujoco_planar_snake/HP_Results.xlsx'

    updates = pd.DataFrame({'lr': [hparams['lr']],
                              'epochs': [hparams['epochs']],
                              'batch_size': [hparams['batch_size']],
                              'dataset_size': [hparams['dataset_size']],
                              'loss': [results['loss']],
                              'accuracy': [results['accuracy']],
                              'time':[ctime()]

                            })


    old_excel_file = pd.read_excel(excel_file_path, index=False)

    excel_file = old_excel_file.append(updates)

    writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')

    excel_file.to_excel(writer, index=False, sheet_name='main')

    writer.save()