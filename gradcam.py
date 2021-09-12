import jax.numpy as jnp
import jax

def resnet(resnet, conv_out, test_local_stats=False):
  print(conv_out.shape, resnet.conv_to_logits(conv_out).shape)
  jac = jax.jacfwd(resnet.conv_to_logits)(conv_out)
  
  importance_weights = jnp.mean(jac, axis=[1, 2]) # Global average pooling
  cam = jax.nn.relu(jnp.dot(importance_weights, conv_out))
  
  return cam