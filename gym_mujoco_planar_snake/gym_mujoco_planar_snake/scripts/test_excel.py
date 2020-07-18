import pandas as pd
import os
import xlwt

def v1():

    hparams = {
        "lr": 0.001,
        "epochs": 50,
        "batch_size": 64,
        "dataset_size": 1000
    }

    results = {
        "loss": 0.001,
        "accuracy": 0.98
    }

    df_marks = pd.DataFrame({'name': ['Somu', 'Kiku', 'Amol', 'Lini'],
                             'physics': [68, 74, 77, 78],
                             'chemistry': [84, 56, 73, 69],
                             'algebra': [78, 88, 82, 87]})

    df_marks2 = pd.DataFrame({'lr': [hparams['lr']],
                             'epochs': [hparams['epochs']],
                             'batch_size': [hparams['batch_size']],
                             'dataset_size': [hparams['dataset_size']],
                             'loss': [results['loss']],
                             'accuracy':[results['accuracy']] })

    df_marks = df_marks.append(df_marks2)



    excel_file_path = os.getcwd() + '/gym_mujoco_planar_snake/HP_Results.xlsx'

    excel_file = pd.read_excel(excel_file_path, index=False)


    #index = len(excel_file["Learning Rate"]) - 1


    #excel_file["Learning Rate"][9] = 46

    excel_file = excel_file.append(df_marks2)

    #l = excel_file["Learning Rate"]


    writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')
    excel_file.to_excel(writer, index=False, sheet_name='report')


    writer.save()
    # l.to_excel('output.xlsx', index=False)

v1()

