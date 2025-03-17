"""
Create the .yaml for each experiment
"""
import os


def create_config(cfg):
    cfg["save_name"] = "{alg}_{dataset}_{num_lb}_{seed}".format(
        alg=cfg["algorithm"],
        dataset=cfg["dataset"],
        num_lb=cfg["num_labels"],
        seed=cfg["seed"],
    )

    cfg["resume"] = True
    cfg["load_path"] = "{}/{}/latest_model.pth".format(
        cfg["save_dir"], cfg["save_name"]
    )

    # Path to store the generated configuration files;
    alg_file = "../config/usb_cv/" + cfg["algorithm"] + "/"
    if not os.path.exists(alg_file):
        os.mkdir(alg_file)

    print(alg_file + cfg["save_name"] + ".yaml")
    with open(alg_file + cfg["save_name"] + ".yaml", "w", encoding="utf-8") as w:
        lines = []
        for k, v in cfg.items():
            line = str(k) + ": " + str(v)
            lines.append(line)
        for line in lines:
            w.writelines(line)
            w.write("\n")


def prepare_configs(
        algorithm,
        save_dir,
        load_path,
        img_size,
        crop_ratio,
        dataset,
        num_classes,
        num_labels,
        net,
        pretrain_path,
        lr,
        weight_decay,
        layer_decay,
        epoch,
        uratio,
        port,
        gpu,
        seed,
        warmup=5,
):
    cfg = {}
    # USB basic arguments;
    cfg["algorithm"] = algorithm
    cfg["amp"] = False
    cfg["use_cat"] = True

    # Save config;
    cfg["save_dir"] = save_dir
    cfg["save_name"] = ""
    cfg["resume"] = False
    cfg["load_path"] = load_path
    cfg["overwrite"] = True
    cfg["use_tensorboard"] = False
    cfg["use_wandb"] = True
    cfg["use_aim"] = False
    cfg["num_log_iter"] = 256

    # Algorithm arguments: FixMatch;
    cfg["hard_label"] = True
    cfg["T"] = 0.5
    cfg["p_cutoff"] = 0.95
    cfg["ulb_loss_ratio"] = 1.0
    # Algorithm arguments: UA;
    cfg["rescale"] = True
    cfg["reweight"] = True
    cfg["p_model_momentum"] = 0.999
    cfg["p_target_momentum"] = 0.99999
    cfg["p_target_type"] = "uniform"

    # Training arguments: Data;
    cfg["img_size"] = img_size
    cfg["crop_ratio"] = crop_ratio
    cfg["data_dir"] = "/data/datasets/usb"
    cfg["dataset"] = dataset
    cfg["num_classes"] = num_classes
    cfg["num_labels"] = num_labels

    # Training arguments: Model;
    cfg["net"] = net
    cfg["net_from_name"] = False
    cfg["use_pretrain"] = True
    cfg["pretrain_path"] = pretrain_path

    # Training arguments: Optimizer;
    cfg["optim"] = "AdamW"
    cfg["lr"] = lr
    cfg["momentum"] = 0.9
    cfg["weight_decay"] = weight_decay
    cfg["layer_decay"] = layer_decay
    cfg["num_warmup_iter"] = int(1024 * warmup)
    cfg["clip"] = 0.0

    # Training arguments: Trainer;
    cfg["epoch"] = epoch
    cfg["num_train_iter"] = 1024 * epoch
    cfg["train_sampler"] = "RandomSampler"
    cfg["uratio"] = uratio
    cfg["batch_size"] = 8

    # Evaluation arguments;
    cfg["num_eval_iter"] = 2048
    cfg["eval_batch_size"] = 128
    cfg["ema_m"] = 0.0

    # Training arguments: GPU and Multi-processing;
    # For NLP num_worker needs handling;
    cfg["num_workers"] = 4
    cfg["world_size"] = 1
    cfg["rank"] = 0
    cfg["multiprocessing_distributed"] = False
    cfg["dist_url"] = "tcp://127.0.0.1:" + str(port)
    cfg["dist_backend"] = "nccl"
    cfg["gpu"] = gpu

    # Randomness;
    cfg["seed"] = seed

    return cfg


# prepare the configuration for baseline model, use_penalty == False
def create_configs(algorithm, size='s', seeds=None, gpus=None):
    # Mostly fixed arguments;
    save_dir = "/data/yuwang/workspace/usb_cv/{}".format(algorithm)
    epoch = 200
    dist_port = range(10001, 11120, 1)
    count = 0
    pretrain_path = "https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0"  # noqa: E501
    weight_decay = 5e-4
    load_path = None

    # Changeable arguments;
    datasets = [
        "cifar100",
        # "stl10",
        # "eurosat",
        # "tissuemnist",
        # "semi_aves"
    ]

    if seeds is None:
        seeds = [0, 1, 2]

    if gpus is None:
        gpus = [0, 1, 2]

    for dataset in datasets:
        for seed, gpu in zip(seeds, gpus):
            if dataset == "cifar100":
                num_classes = 100
                if size == 's':
                    num_labels = 200
                else:
                    num_labels = 400

                img_size = 32
                crop_ratio = 0.875
                net = "vit_small_patch2_32"
                pretrain_name = "vit_small_patch2_32_mlp_im_1k_32.pth"

                lr = 5e-4
                layer_decay = 0.5
                uratio = 1

            elif dataset == "stl10":
                num_classes = 10
                if size == 's':
                    num_labels = 40
                else:
                    num_labels = 100
                img_size = 96
                crop_ratio = 0.875

                net = "vit_base_patch16_96"
                pretrain_name = "mae_pretrain_vit_base.pth"

                lr = 1e-4
                layer_decay = 0.65
                uratio = 1

            elif dataset == "eurosat":
                num_classes = 10
                if size == 's':
                    num_labels = 20
                else:
                    num_labels = 40

                img_size = 32
                crop_ratio = 0.875

                net = "vit_small_patch2_32"
                pretrain_name = "vit_small_patch2_32_mlp_im_1k_32.pth"

                lr = 5e-5
                layer_decay = 1.0
                uratio = 1

            elif dataset == "tissuemnist":
                num_classes = 8
                if size == 's':
                    num_labels = 80
                else:
                    num_labels = 400

                img_size = 32
                crop_ratio = 0.95

                net = "vit_tiny_patch2_32"
                pretrain_name = "vit_tiny_patch2_32_mlp_im_1k_32.pth"

                lr = 5e-5
                layer_decay = 0.95
                uratio = 1

            elif dataset == "semi_aves":
                num_classes = 200
                num_labels = 3959

                img_size = 224
                crop_ratio = 0.875

                net = "vit_small_patch16_224"
                pretrain_name = "vit_small_patch16_224_mlp_im_1k_224.pth"

                lr = 1e-3
                layer_decay = 0.65
                uratio = 1

            else:
                raise ValueError('Not a valid dataset name, get{}'.format(dataset))

            # Prepare the configuration file;
            cfg = prepare_configs(
                algorithm,
                save_dir,
                load_path,
                img_size,
                crop_ratio,
                dataset,
                num_classes,
                num_labels,
                net,
                os.path.join(pretrain_path, pretrain_name),
                lr,
                weight_decay,
                layer_decay,
                epoch,
                uratio,
                dist_port[count],
                gpu,
                seed,
                warmup=5,
            )
            count += 1

            create_config(cfg)


def create_scripts(algorithm):
    config_path = "config/usb_cv/{}".format(algorithm)

    with open('../exp_run.sh', 'w') as file:
        file.write("#!/bin/bash\n")  # Optional: Starting line for a bash script

        # Loop through each file in the directory
        for filename in os.listdir("../" + config_path):
            # Check if it is a file and not a directory
            if os.path.isfile(os.path.join(config_path, filename)):
                # Write the command line to the script file
                file.write(f"python train.py --c {config_path}/{filename}\n")


if __name__ == "__main__":
    print("Generating the following configuration files and corresponding scripts...")
    create_configs("tamatch")

    print("Generating the scripts..")
    create_scripts("tamatch")

    print("Finished all tasks.")



