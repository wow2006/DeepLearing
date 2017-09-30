from tflearn.layers.core import dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.conv import conv_2d, max_pool_2d

def createArchitecture(archDiscription, inputLayer):
    funcDic = {"conv_2d":                      conv_2d,
               "max_pool_2d":                  max_pool_2d,
               "dropout":                      dropout,
               "fully_connected":              fully_connected,
               "local_response_normalization": local_response_normalization}
    
    network = inputLayer
    for item in archDiscription:
        for key in item:
            network = funcDic[key](network, **item[key])
        
    return network
