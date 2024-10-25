<p align="center">
   <img src="Logo/logo.png" width="50%" align='center' />
</p>


# FCS-HGNN: Flexible Multi-type Community Search in Heterogeneous Information Networks (CIKM 2024)

This is the repository for our research paper "[FCS-HGNN: Flexible Multi-type Community Search in Heterogeneous Information Networks](https://dl.acm.org/doi/abs/10.1145/3627673.3679696)".

> Guoxin Chen, Fangda Guo, Yongqing Wang, Yanghao Liu, Peiying Yu, Huawei Shen, and Xueqi Cheng. 2024. FCS-HGNN: Flexible Multi-type Community Search in Heterogeneous Information Networks. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM '24). Association for Computing Machinery, New York, NY, USA, 207–217. https://doi.org/10.1145/3627673.3679696


# Requirements
* Python 3.9
* Ubuntu 22.04
* Python Packages

```
conda create -n fcs_hgnn python=3.9
conda activate fcs_hgnn
pip install -r requirements.txt
```

# Data
The folders `./datasets` and `datasets_two_type` contain the example data for single-type and multi-type communities, respectively. 
```
cd ./datasets  # or ./datasets_two_type
unzip *.zip
```

# Citation
If our work contributes to your research, please acknowledge it by citing our paper. We greatly appreciate your support.

```
@inproceedings{10.1145/3627673.3679696,
    author = {Chen, Guoxin and Guo, Fangda and Wang, Yongqing and Liu, Yanghao and Yu, Peiying and Shen, Huawei and Cheng, Xueqi},
    title = {FCS-HGNN: Flexible Multi-type Community Search in Heterogeneous Information Networks},
    year = {2024},
    isbn = {9798400704369},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3627673.3679696},
    doi = {10.1145/3627673.3679696},
    pages = {207–217},
    numpages = {11},
    keywords = {community search, multi-type community},
    location = {Boise, ID, USA},
    series = {CIKM '24}
}
```