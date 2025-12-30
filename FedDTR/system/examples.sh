#!/bin/bash

# ===============================================================horizontal(mnist)======================================================================

# rm ../dataset/mnist/config.json
# cd ../dataset/
# nohup python -u generate_mnist.py noniid - dir > mnist_dataset.out 2>&1 #dir
# cd ../system/

nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m cnn -algo FedGhp -gr 2000 -did 0 -go cnn > ../result/mnist_Fedghp_pat.out 2>&1 & #test.out 2>&1 &#
