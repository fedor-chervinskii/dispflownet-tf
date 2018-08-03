# dispflownet-tf

Reimplementation in tensorflow of Dispnet and Dispnet-Corr1D.
Original code thx to [@fedor-chervinskii](https://github.com/fedor-chervinskii/dispflownet-tf)

## Improvement and fixes
+ Python3 conversion
+ Add compile.sh inside user_ops
+ Modified image preprocessing to match those used in caffe
+ Added implementation of ["Unsupervised Adaptation for Deep Stereo" - ICCV2017](https://github.com/CVLAB-Unibo/Unsupervised-Adaptation-for-Deep-Stereo) - from now on shortened 'UA'
+ Lots of arguments from command line
+ Native tensorflow input pipeline
+ Add improved code for inference

## Pretrained nets
+ Weights for Dispnet with tensorflow correlation layer trained for 1200000 step on FlyingThings3D available [here](https://drive.google.com/open?id=1BAxWAOghm0DTeXMQZQJuTPBCV5PtRPMt) 

## Training
1. Create a training and validation set made of couple of left and right frames + disparities (+ confidence for UA)
2. Create a txt file with the list of training samples for your trainign set, each row on the file should contain "path_left_frame;peth_right_frame;path_ground_truth", for UA "path_left_frame;peth_right_frame;path_disparity;path_confidence". For UA "path_disparity" is the path to the disparity map obtained by a standard, non learned, stereo algorithm (e.g.: SGM or AD-CENSUS in the paper)
3. Train Dispnet using train.py, usage:
```
python main.py --training $TRAINING_LIST --testing $TEST_LIST -c $OUT_DIR --corr_type tf 
```
Arguments, some are optional or already provide a default value:
+ --training **TRAININGLIST**: path to the training list as defined at 2.
+ --testing **TEST_LIST**: path to the test list as defined at 2. 
+ -c **CHECKPOINT_PATH**: path were the log and trained CNN will be saved.
+ -b **BATCH_SIZE**: number of samples for each train iteration.
+ -l **LOG_STEP**: every how many step save summaries.
+ -w **FILE**: optional initialization weights.
+ -s **SAVE_STEP**: every how many step save the current network.
+ -n **N_STEP**: number of training step to perform
+ --corr_type **[tf,cuda,none]**: type of correlation layer implemented in dispnet ('tf'->pure tensorflow implementation, 'cuda'->native cuda implementation, needs to manually compile the op defined in user_ops/, 'none'->no correlation layer)
+ --kittigt: load gt map as 16bit png image with each pixel encoding=disparity x 256

Additional arguments for UA:
+ -th **CONFIDENCE_TH**: minimum confidence value to consider a gt pixel as valid
+ --smooth **SMOOTH**: multiplier for the smoothing twerm of the loss function
+ --doubleConf: flag to read confidence map as 16bit png image with each pixel encoding=confidence x 256 x 256

## Test/inference
inference.py can be used to perform inference with any kind of Dispnet, even the one trained using UA, on a list of stereo frames and save the resulting disparities on disk.
```
python inference.py --left $LEFT_FOLDER --right $RIGHT_FOLDER --ckpt $CKPT_PATH -o $OUT_FOLDER --fullRes -v
```
Arguments, some are optional or already provide a default value:
+ --left **LEFT**: either path to a folder containing the left frames or to a txt file with the list of left frame paths to load.
+ --right **RIGHT**: either path to a folder containing the right frames or to a txt file with the list of left frame paths to load.
+ --ckpt **CKPT**: path to the weight of Dispnet that needs to be loaded.
+ --corr_type **[tf,cuda,none]**: type of correlation layer implemented in dispnet ('tf'->pure tensorflow implementation, 'cuda'->native cuda implementation, needs to manually compile the op defined in user_ops/, 'none'->no correlation layer)
+ -o **OUT_DIR**: path to the output dir where the results will be saved
+ -v: flag to enable visualization
+ --fullRes: flag to save the output of the network rescaled at the full input resolution
+ --max_disp **MAX**: maximum value for disparity, value above will be clipped before saving the disparities.

