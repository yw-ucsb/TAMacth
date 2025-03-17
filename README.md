## TAMatch (Towards the Mitigation of Confirmation Bias in Semi-supervised Learning: a Debiased Training Perspective)

Temporary code repo to check the implementations of **Towards the Mitigation of Confirmation Bias in Semi-supervised Learning: a Debiased Training Perspective**


- To install dependency: check the [USB repo](https://github.com/microsoft/Semi-supervised-learning).
- To generate config files: check [TAMatch config generator](https://github.com/yw-ucsb/TAMacth/blob/tamatch/scripts/tamatch_config_script_generator_usb_cv.py).
- To setup the wandb logger (required), check https://wandb.ai/site/ and setup the corresponding environment variables.
- To run the code: run ```bash ./exp_run.sh```.

TAMatch builds upon USB. Some critical implementations details can be found in these directories:

- Implementation of the [algorithm](https://github.com/yw-ucsb/TAMacth/blob/tamatch/semilearn/algorithms/tamatch/tamatch.py).
- Implementation of the [debias hook](https://github.com/yw-ucsb/TAMacth/blob/tamatch/semilearn/algorithms/tamatch/utils.py).
