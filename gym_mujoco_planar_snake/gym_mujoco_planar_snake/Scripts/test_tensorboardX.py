from tensorboardX import SummaryWriter

writer = SummaryWriter("../log/tensorboard/v1")

for n_iter in range(100):


    # data grouping by `slash`
    writer.add_scalar('data/scalar1',  n_iter)

writer.close()
