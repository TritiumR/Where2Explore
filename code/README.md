# Experiments
This folder includes the codebase for Where2Explore simulator and experiments.

## Before start
To train the models, please first go to the `../data` folder and download the pre-processed SAPIEN dataset for Where2Explore. 

To test over the pretrained models, please go to the `logs` folder and download the pretrained checkpoints.

Please fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSegEvIM22Ta44MrKM5d-guRE4aDR5K77ZQoInLWEyib-aeCFw/viewform?usp=sf_link) to download all resources.

## Dependencies
This code has been tested on Ubuntu 18.04 with Cuda 10.1, Python 3.6, and PyTorch 1.7.0.

First, install SAPIEN following

    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl

For other Python versions, you can use one of the following

    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp35-cp35m-manylinux2014_x86_64.whl
    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp37-cp37m-manylinux2014_x86_64.whl
    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp38-cp38-manylinux2014_x86_64.whl

Please do not use the default `pip install sapien` as SAPIEN is still being actively developed and updated.

Then, if you want to run the 3D experiment, this depends on PointNet++.

    git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
    cd Pointnet2_PyTorch
    # [IMPORTANT] comment these two lines of code:
    #   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
    pip install -r requirements.txt
    pip install -e .

Finally, run the following to install other packages.
   
    # make sure you are at the repository root directory
    pip install -r requirements.txt

to install the other dependencies.

For visualization, please install blender v2.79 and put the executable in your environment path.
Also, the prediction result can be visualized using MeshLab or the *RenderShape* tool in [Thea](https://github.com/sidch/thea).

## Simulator
You can run the following command to test and visualize a random interation in the simulation environment.

    python collect_data.py 40147 StorageFurniture 0 pushing

Change the shape id to other ids for testing other shapes, 
and modify the primitive action type to any of the six supported types: *pushing, pushing-up, pushing-left, pulling, pulling-up, pulling-left*. 
Run `python collection_data.py --help` to understand the full input arguments. 

After you ran the code, you will find a record for this interaction trial under `./results/40147_StorageFurniture_0_pushing_0`, from where you can see the full log, 2D image, 3D depth and interaction outcome.
You can run the following command to replay the interaction.

    python replay_data.py results/40147_StorageFurniture_0_pushing_0/result.json

If you want to run on a headless server, simple put `xvfb-run -a ` before any code command that runs the SAPIEN simulator.
Install the `xvfb` tool on your server if not installed.

## Generate Offline Training Data
Before training the network, we need to collect a large set of interaction trials via random exploration, using the 
script `scripts/run_gen_offline_data_pushing.sh` and `scripts/run_gen_offline_data_pulling.sh`.

Each script will generate 3 type of data: interactions on training categories (StorageFurniture,Faucet,Window) for training; 
object shapes from novel categories (11 categories) for few-shot learning; and interactions on novel categories (11 categories) for testing.

To generate data for pushing experiments, please run:

```
bash scripts/run_gen_offline_data_pushing.sh
```

To generate data for pulling experiments, please run:

    bash scripts/run_gen_offline_data_pulling.sh

## Experiment
To train the network, first train the model on training,

    bash scripts/run_train_3d_explore_pushing.sh
    bash scripts/run_train_3d_explore_pulling.sh

then, conduct few-shot learning on novel categories,

    bash scripts/run_train_few-shot_pushing.sh
    bash scripts/run_train_few-shot_pulling.sh

To evaluate the model on unseen objects from novel categories, run

    bash scripts/run_test_pushing.sh
    bash scripts/run_test_pulling.sh
    
To visualize the affordance and similarity prediction on a specific object, please run

    bash scripts/run_visualize.sh

and change the 'list' into the object id you want to visualize.
