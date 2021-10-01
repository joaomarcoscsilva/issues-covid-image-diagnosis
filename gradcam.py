import jax.numpy as jnp
import jax
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import utils

def resnet(resnet, conv_out, counterfactual):
  jac = jax.jacrev(resnet.conv_to_logits)(conv_out)

  batch_size = conv_out.shape[0]
  num_classes = jac.shape[1]
  num_features = jac.shape[-1]

  assert batch_size == 1

  if counterfactual:
    jac *= -1

  importance_weights = jnp.mean(jac, axis=[-2, -3]) # Global average pooling
  importance_weights = importance_weights.reshape(num_classes, num_features)
  conv_out = conv_out.reshape(conv_out.shape[1:])

  output = []
  for class_index in range(num_classes):
    cam = conv_out.dot(importance_weights[class_index,:])
    cam = jax.nn.relu(cam)
    cam /= cam.max()
    output.append(cam)

  output = jnp.stack(output)

  return output

def generate_from_data(x_gradcam_data, y_gradcam_data, net, params, state, gradcam_counterfactual=False):
  gradcam_batch_size = 8
  datagen, _ = dataset.get_datagen(True, gradcam_batch_size, x_gradcam_data, y_gradcam_data, include_last = False)

  # List with all the predictions
  cams = []

  # Applies the network to each batch
  for x_batch, _ in tqdm(datagen()):
      apply = jax.pmap(net.apply, static_broadcasted_argnums = (2,4,5,6,7))

      cams.append(apply(params, state, None, x_batch, True, False, True, gradcam_counterfactual)[0])

  cams = jnp.concatenate(cams)
  return cams

def plot_heatmap(img, heatmap, alpha=0.4):
  """Plots a single Grad-CAM with a Jet colormap.

  Args:
      img (np.array): Image to display.
      heatmap (np.array): The overlay heatmap.
      alpha (float, optional): Alpha of the blending. Defaults to 0.4.
  """
  # Rescale heatmap to a range 0-255
  heatmap /= heatmap.max()
  heatmap = np.uint8(255 * heatmap)

  # Use jet colormap to colorize heatmap
  jet = cm.get_cmap("jet")

  # Use RGB values of the colormap
  jet_colors = jet(np.arange(256))[:, :3]
  jet_heatmap = jet_colors[heatmap]
  jet_heatmap = cv2.resize(jet_heatmap, dsize=(
      img.shape[0], img.shape[1]), interpolation=cv2.INTER_LINEAR)
  jet_heatmap /= jet_heatmap.max()
  jet_heatmap = np.clip(jet_heatmap, 0, 1)

  # Superimpose the heatmap on original image
  superimposed_img = jet_heatmap * alpha + img
  superimposed_img /= superimposed_img.max()
  # Display Grad CAM
  plt.imshow(superimposed_img)

def plot_gradcams_from_class(target_class, x_gradcam_data, y_gradcam_data, y_pred, cams, rows=3, columns=8, rng=jax.random.PRNGKey(987987689)):
  """
  Plots a sample of the Grad-CAMs from images of a certain class. The Grad-CAM displayed is dependant on the prediction of the model for each image.

  Args:
      target_class (int): Class index to display.
      x_gradcam_data (np.array): X data used to generate the Grad-CAM.
      y_gradcam_data (np.array): Y data used to generate the Grad-CAM.
      y_pred (np.array): Y prediction of the model for the images in x_gradcam_data.
      cams (np.array): Calculated Grad-CAMs (could be either counter cams or the actual grad-cams).
      rows (int, optional): Number of rows of images to display. Defaults to 3.
      columns (int, optional): Number of columns of images to display. Defaults to 8.
      rng (jax.random.PRNGKey, optional): Defaults to jax.random.PRNGKey(987987689).
  """
  real = jnp.argmax(y_gradcam_data, axis=1)
  predicted = jnp.argmax(y_pred, axis=1)
  assert predicted.shape[0] == real.shape[0] and real.shape[0] == y_gradcam_data.shape[0]

  indices_of_class = jnp.where(real == target_class)[0]

  print("Perc correct {:.4f}".format(
      jnp.mean((predicted[indices_of_class] == target_class).astype(jnp.float32))))

  indices_of_class = jax.random.choice(rng, indices_of_class, shape=[
      rows*columns], replace=False)

  fig = plt.figure(figsize=(32, 12))
  fig.suptitle(utils.CLASS_NAMES[target_class] + " - Grad-CAM", fontsize=16)

  for i in range(rows*columns):
      fig.add_subplot(rows, columns, i+1)

      img_index = indices_of_class[i]
      pred = predicted[img_index]
      img = x_gradcam_data[img_index, ]
      heatmap = cams[img_index, pred, ]

      plot_heatmap(img, heatmap, alpha=0.5)

      title = "Pred: " + utils.CLASS_NAMES[pred]

      plt.title(title)
      plt.axis('off')
