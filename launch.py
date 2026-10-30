# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg):
    if cfg.mode == "eval":
        from evaluate import evaluate

        return evaluate(cfg)
    else:
        from training.trainer import Trainer

        trainer = Trainer(**cfg)
        trainer.run()


if __name__ == "__main__":
    main()
