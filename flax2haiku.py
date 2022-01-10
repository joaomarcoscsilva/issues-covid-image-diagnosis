from jax_resnet import pretrained_resnet
from jax import numpy as np
import jax
import resnet
import haiku as hk

def load_imagenet_resnet(params):

    hk_params = {}

    ResNet, flax_variables = pretrained_resnet(size = 18)

    def bn(flax_dict):
        return {'scale' : flax_dict['scale'].reshape(1,1,1,-1),
                'offset' : flax_dict['bias'].reshape(1,1,1,-1)}

    def dense(flax_dict):
        return {'w' : flax_dict['kernel']}

    def convlayer(flax_dict):
        return dense(flax_dict['Conv_0']), bn(flax_dict['BatchNorm_0'])

    hk_params['res_net18/~/initial_conv'] = dense(flax_variables['params']['layers_0']['ConvBlock_0']['Conv_0'])
    hk_params['res_net18/~/initial_batchnorm'] = bn(flax_variables['params']['layers_0']['ConvBlock_0']['BatchNorm_0'])

    for i in range(16):
        s = f'res_net18/~/block_group_{(i//4)}/~/block_{(i//2)%2}/~/'
        conv = s + f'conv_{i%2}'
        batchnorm = s + f'batchnorm_{i%2}'
        hk_params[conv], hk_params[batchnorm] = convlayer(flax_variables['params'][f'layers_{2 + i // 2}'][f'ConvBlock_{i % 2}'])
        

    for i in range(3):
        s = f'res_net18/~/block_group_{i+1}/~/block_0/~/'
        conv = s + 'shortcut_conv'
        batchnorm = s + 'shortcut_batchnorm'
        hk_params[conv], hk_params[batchnorm] = convlayer(flax_variables['params'][f'layers_{4 + 2*i}']['ResNetSkipConnection_0']['ConvBlock_0'])

    hk_params = jax.device_put_replicated(hk_params, jax.local_devices())
    
    hk_params['res_net18/~/logits'] = params['res_net18/~/logits']

    hk_params = hk.data_structures.to_immutable_dict(hk_params)
    return hk_params