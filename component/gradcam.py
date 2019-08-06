"""
This could was taken from the author below.
It has been customized to fix my model 

@author: Utku Ozbulak - github.com/utkuozbulak
@original code: https://github.com/utkuozbulak/pytorch-cnn-visualizations
"""
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import os

import matplotlib.cm as mpl_color_map
import copy

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None

        for pos, layer in enumerate(self.model.resnet):
            x = x.cuda()
            x = layer(x)  # Forward
            if int(pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        # x = self.model.classifier(x)
        # x = self.model.classifier(x)

        # x = x.view(x.size(0), -1)  # Flatten
        # x = x[0]
        x = x.reshape(x.size(0), -1)
        # print("self.model.linear", self.model.linear.shape)
        # features = self.bn(self.linear(features))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.cuda()
        x = x.to(device)

        self.model = self.model.cuda()
        self.model = self.model.to(device)

        x = self.model.linear(x)
        x = self.model.bn(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        # self.model.features.zero_grad()
        # self.model.fc.in_features.zero_grad()
        # self.model.classifier.zero_grad()
        self.model.resnet.zero_grad()
        self.model.linear.zero_grad()
        self.model.bn.zero_grad()

        one_hot_output = one_hot_output.cuda()
        one_hot_output = one_hot_output.to(device)
        # model_output.cuda()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.
        return cam

    def save_class_activation_images(self, org_img, activation_map, file_name):
        """
            Saves cam activation map and activation map on the original image
        Args:
            org_img (PIL img): Original image
            activation_map (numpy arr): Activation map (grayscale) 0-255
            file_name (str): File name of the exported image
        """
        # Grayscale activation map
        heatmap, heatmap_on_image = self.apply_colormap_on_image(org_img, activation_map, 'hsv')
        # Save colored heatmap
        path_to_file = os.path.join(file_name+'_Cam_Heatmap.png')
        self.save_image(heatmap, path_to_file)
        # Save heatmap on iamge
        path_to_file = os.path.join(file_name+'_Cam_On_Image.png')
        self.save_image(heatmap_on_image, path_to_file)
        # SAve grayscale heatmap
        path_to_file = os.path.join(file_name+'_Cam_Grayscale.png')
        self.save_image(activation_map, path_to_file)

    def preprocess_image(self, pil_im):
        """
            Processes image for CNNs
        Args:
            PIL_img (PIL_img): Image to process
            resize_im (bool): Resize to 224 or not
        returns:
            im_as_var (torch variable): Variable that contains processed float tensor
        """
        # mean and std list for channels (Imagenet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Resize image
        pil_im.thumbnail((224, 224))
        im_as_arr = np.float32(pil_im)
        im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
        # Normalize the channels
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
            im_as_arr[channel] -= mean[channel]
            im_as_arr[channel] /= std[channel]
        # Convert to float tensor
        im_as_ten = torch.from_numpy(im_as_arr).float()
        # Add one more channel to the beginning. Tensor shape = 1,3,224,224
        im_as_ten.unsqueeze_(0)
        # Convert to Pytorch variable
        im_as_var = Variable(im_as_ten, requires_grad=True)
        return im_as_var

    def apply_colormap_on_image(self, org_im, activation, colormap_name):
        """
            Apply heatmap on image
        Args:
            org_img (PIL img): Original image
            activation_map (numpy arr): Activation map (grayscale) 0-255
            colormap_name (str): Name of the colormap
        """
        # Get colormap
        color_map = mpl_color_map.get_cmap(colormap_name)
        no_trans_heatmap = color_map(activation)
        # Change alpha channel in colormap to make sure original image is displayed
        heatmap = copy.copy(no_trans_heatmap)
        heatmap[:, :, 3] = 0.4
        heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
        no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

        # Apply heatmap on iamge
        heatmap_on_image = Image.new("RGBA", org_im.size)
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
        return no_trans_heatmap, heatmap_on_image

    def save_image(self, im, path):
        """
            Saves a numpy matrix or PIL image as an image
        Args:
            im_as_arr (Numpy array): Matrix of shape DxWxH
            path (str): Path to the image
        """
        if isinstance(im, (np.ndarray, np.generic)):
            im = self.format_np_output(im)
            im = Image.fromarray(im)
        im.save(path)


    def format_np_output(self,np_arr):
        """
            This is a (kind of) bandaid fix to streamline saving procedure.
            It converts all the outputs to the same format which is 3xWxH
            with using sucecssive if clauses.
        Args:
            im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
        """
        # Phase/Case 1: The np arr only has 2 dimensions
        # Result: Add a dimension at the beginning
        if len(np_arr.shape) == 2:
            np_arr = np.expand_dims(np_arr, axis=0)
        # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
        # Result: Repeat first channel and convert 1xWxH to 3xWxH
        if np_arr.shape[0] == 1:
            np_arr = np.repeat(np_arr, 3, axis=0)
        # Phase/Case 3: Np arr is of shape 3xWxH
        # Result: Convert it to WxHx3 in order to make it saveable by PIL
        if np_arr.shape[0] == 3:
            np_arr = np_arr.transpose(1, 2, 0)
        # Phase/Case 4: NP arr is normalized between 0-1
        # Result: Multiply with 255 and change type to make it saveable by PIL
        if np.max(np_arr) <= 1:
            np_arr = (np_arr*255).astype(np.uint8)
        return np_arr



if __name__ == '__main__':
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')
