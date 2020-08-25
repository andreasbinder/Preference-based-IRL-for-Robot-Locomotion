import yaml

path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/agents/configurations/configs.yml"

with open(path) as file:
    documents = yaml.load(file)

    '''for item, doc in documents.items():
        print(item, ":", doc)'''

    #documents.update({'reward_learning': {'hparams': {'lr': 5}}})

    #yaml.dump(documents)

'''with open(path, 'w') as f:
    data = yaml.dump(documents, f)
    print(documents)'''