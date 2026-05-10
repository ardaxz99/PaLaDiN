from header import *
from datasets import *
from model import *
from config import *
import matplotlib.pyplot as plt

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--log_path', type=str)
    # model configurations

    parser.add_argument('--stage', type=int) # the maximum sequence length
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory')

    return parser.parse_args()

def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def config_env(args):
    args['root_dir'] = '../'
    args['mode'] = 'train'
    config = load_config(args)
    

    print(f"[INFO] Loaded config: {config}")
    
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])

def build_directory(path):
    if os.path.exists(path):
        pass
    else: # recursively construct directory
        os.makedirs(path, exist_ok=True)

def main(**args):
    config_env(args)
    args['ds_config_path'] = f'dsconfig/deepspeed_paladin_stage{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])
    build_directory(args['log_path'])

    train_data, train_iter, sampler = load_visa_dataset(args)

    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = 2 * args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()
    agent.save_model(args['save_path'])

    # begin to train
    pbar = tqdm(total= 2 * length, disable=True)    # maximum total number
    current_step = 0
    epoch_mean_losses = []

    for epoch_i in tqdm(range(args['epochs']), disable=True):
        iter_every_epoch = 0
        epoch_losses = []

        for step_in_epoch, batch in enumerate(train_iter):

            iter_every_epoch += 1

            loss = agent.train_model(
                batch, 
                current_step=current_step, 
                pbar=pbar
            )
            epoch_losses.append(loss)
            del batch
            current_step += 1

        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_mean_losses.append(mean_epoch_loss)
        print(f"\n✅ [Epoch {epoch_i+1} completed] Average Loss: {round(mean_epoch_loss, 4)}\n")

        torch.distributed.barrier()
        if (epoch_i + 1) % 10 == 0:
            save_path = f"{args['save_path']}/epoch_{epoch_i+1}"
            build_directory(save_path)  # ensure the directory exists
            agent.save_model(save_path)
            
    # 📊 Plot the epoch losses
    plt.figure()
    plt.plot(range(1, len(epoch_mean_losses) + 1), epoch_mean_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)

    plot_path = os.path.join(args['log_path'], 'epoch_loss_plot.png')
    plt.savefig(plot_path)
    print(f"[📉] Saved training loss plot at: {plot_path}")
    plt.close()


if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
