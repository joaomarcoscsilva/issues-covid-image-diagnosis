import jax.numpy as jnp
import jax

def resnet(resnet, conv_out, counterfactual):
  print("Convout", conv_out.shape)
  print(conv_out)
  jac = jax.jacrev(resnet.conv_to_logits)(conv_out)
  print("jac", jac.shape)

  batch_size = conv_out.shape[0]
  num_classes = jac.shape[1]
  num_features = jac.shape[-1]

  assert batch_size == 1

  if counterfactual:
    jac *= -1

  importance_weights = jnp.mean(jac, axis=[-2, -3]) # Global average pooling
  print("BF", importance_weights.shape)
  importance_weights = importance_weights.reshape(num_classes, num_features)
  print("AF", importance_weights.shape)
  conv_out = conv_out.reshape(conv_out.shape[1:])

  output = []
  for class_index in range(num_classes):
    print(conv_out.shape, "*", importance_weights[class_index,:].shape)
    cam = conv_out.dot(importance_weights[class_index,:])
    cam = jax.nn.relu(cam)
    cam /= cam.max()
    print(cam.shape)
    output.append(cam)

  output = jnp.stack(output)

  return output