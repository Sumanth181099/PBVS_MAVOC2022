

# Multi-Modal Domain Fusion for Multi-modal Aerial View Object Classification

In this work, we explore a methodology to use both EO and SAR sensor’s information to effectively improve the performance of the ATR (Automatic Target Recognition) systems by handling the shortcomings of each of the sensors. A novel Multi-Modal Domain Fusion(MDF) network is proposed to learn the domain invariant features from multi-modal data and use it to accurately classify the aerial view objects.  The proposed MDF network achieves top-10 performance in the Track-1 (test on SAR data only) with an accuracy of 25.3% and top-5 performance in Track-2 (test on EO and SAR data) with an accuracy of 34.26% in the test phase on the PBVS MAVOC Challenge dataset.

Check out our paper [here](https://arxiv.org/pdf/2212.07039.pdf).


We advise you to use conda environment to run the package. Run the following command to install all the necessary modules:

```sh
conda env create -f environment.yaml 
conda activate PBVS
```

To reproduce our results, you will need to download the [NTIRE/PBVS MAVOC challenge dataset](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/papers/Low_Multi-Modal_Aerial_View_Object_Classification_Challenge_Results_-_PBVS_2022_CVPRW_2022_paper.pdf).

To reproduce the results that we have achieved, please follow the following steps:

0) It is advised that the test data, the test executable file(result_gen.py), the trained model file(.pth file) and the results.csv file are all in the working directory so that path issues do not occur. Now, conda activate <environment_name>. 

1) Replace the path defined in the img_folder1 variable in the result_gen.py file with the path of the EO test images.

2) Replace the path defined in the img_folder2 variable in the result_gen.py file with the path of the SAR test images.

3) Create empty results.csv file with just the headers "image_id" and "class_id" in the working directory.

4) Just execute the result_gen.py using command: python result_gen.py

The results.csv wil be filled with the predictions which you may use to calculate the test accuracy. Thats it to reproduce our results!

The pretrained model ckpt can be found [here](https://drive.google.com/file/d/1RezUvwyLm1S9IEI03x2XrgRFouBQ9PIP/view?usp=drive_link).

For Track1 (SAR only), you will find the model ckpts [here](https://drive.google.com/file/d/1I2Nr5Zzb1uQtVWw9Tho_uXpo02rZZvNI/view?usp=drive_link) and [here](https://drive.google.com/file/d/1DKof7XqpNaierVsTNwkeqF6OhQNmds0_/view?usp=drive_link).

For Track2 (EO and SAR), you will find the model ckpt [here](https://drive.google.com/file/d/10_y7ynJG2Msw3BXGtstwooWuRMQqwXUk/view?usp=drive_link).

## Citation

If you find this repo useful for your work, please cite our paper:

```shell
@misc{udupa2022multimodal,
      title={Multi-Modal Domain Fusion for Multi-modal Aerial View Object Classification}, 
      author={Sumanth Udupa and Aniruddh Sikdar and Suresh Sundaram},
      year={2022},
      eprint={2212.07039},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



