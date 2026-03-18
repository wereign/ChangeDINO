import argparse
import torch
from pathlib import Path


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument(
            "--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0. use -1 for CPU"
        )
        self.parser.add_argument("--name", type=str, default="WHU")
        self.parser.add_argument(
            "--dataroot", type=str, default="/ssddd/chingheng/CD-Dataset"
        )
        self.parser.add_argument("--dataset", type=str, default="WHU-CD")
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="./checkpoints",
            help="models are saved here",
        )
        
        self.parser.add_argument(
            "--save_test", action="store_true"
        )
        self.parser.add_argument(
            "--result_dir", type=str, default="./results", help="results are saved here"
        )
        self.parser.add_argument(
            "--vis_path", type=str, default="vis", help="results are saved here"
        )
        self.parser.add_argument("--load_pretrain", action='store_true')
        self.parser.add_argument("--use_morph", action='store_true')

        self.parser.add_argument("--phase", type=str, default="train")
        self.parser.add_argument("--backbone", type=str, default="mobilenetv2")
        self.parser.add_argument("--fpn", type=str, default="fpn")
        self.parser.add_argument("--fpn_channels", type=int, default=128)
        self.parser.add_argument("--deform_groups", type=int, default=4)
        self.parser.add_argument("--gamma_mode", type=str, default="SE")
        self.parser.add_argument("--beta_mode", type=str, default="contextgatedconv")
        self.parser.add_argument('--n_layers', nargs='+', type=int, default=[1, 1, 1, 1])
        self.parser.add_argument('--extract_ids', nargs='+', type=int, default=[5, 11, 17, 23])
        self.parser.add_argument("--alpha", type=float, default=0.25)
        self.parser.add_argument("--gamma", type=int, default=4, help="gamma for Focal loss")

        self.parser.add_argument("--batch_size", type=int, default=16)
        self.parser.add_argument("--num_epochs", type=int, default=100)
        self.parser.add_argument("--num_workers", type=int, default=4, help="#threads for loading data")
        self.parser.add_argument("--lr", type=float, default=5e-4)
        self.parser.add_argument("--weight_decay", type=float, default=5e-4)

    @staticmethod
    def get_defaults():
        """Get default options as a dictionary for programmatic initialization."""
        return {
            "gpu_ids": [0],
            "name": "WHU",
            "dataroot": "/ssddd/chingheng/CD-Dataset",
            "dataset": "WHU-CD",
            "checkpoint_dir": "./checkpoints",
            "save_test": False,
            "result_dir": "./results",
            "vis_path": "vis",
            "load_pretrain": True,
            "use_morph": False,
            "phase": "train",
            "backbone": "mobilenetv2",
            "fpn": "fpn",
            "fpn_channels": 128,
            "deform_groups": 4,
            "gamma_mode": "SE",
            "beta_mode": "contextgatedconv",
            "n_layers": [1, 1, 1, 1],
            "extract_ids": [5, 11, 17, 23],
            "alpha": 0.25,
            "gamma": 4,
            "batch_size": 16,
            "num_epochs": 100,
            "num_workers": 4,
            "lr": 5e-4,
            "weight_decay": 5e-4,
        }

    @staticmethod
    def create_from_dict(params=None):
        """
        Create an options object programmatically without CLI parsing.
        
        Args:
            params: Dictionary of parameter overrides. Uses defaults for any not specified.
        
        Returns:
            Namespace object with all option attributes set.
        """
        defaults = Options.get_defaults()
        if params:
            defaults.update(params)
        
        # Create a namespace object with all the parameters
        opt = argparse.Namespace(**defaults)
        
        # Resolve checkpoint_dir to absolute path relative to ChangeDINO directory
        changdino_dir = Path(__file__).parent
        if not Path(opt.checkpoint_dir).is_absolute():
            opt.checkpoint_dir = str(changdino_dir / opt.checkpoint_dir)
        
        # Set GPU device if available
        if torch.cuda.is_available() and len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        
        opt.phase = "test"
        opt.load_pretrain = True
        opt.batch_size = 1
        opt.num_workers = 0
        
        return opt

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(",")
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        return self.opt
