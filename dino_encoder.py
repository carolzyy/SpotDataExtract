import torch
from torchvision import transforms
import ssl
import cv2
import numpy as np
class Encoder():
    def __init__(self,device='cuda'):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.device = device
        self.dinov2_vits14 = self.load_dino()

    def load_dino(self,):
        ssl._create_default_https_context = ssl._create_unverified_context
        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        dinov2_vits14.eval()
        dinov2_vits14.cuda()
        return dinov2_vits14



    def image_stich_manul(self,img_r, img_l):
        return cv2.hconcat([img_r[:,40:320,],img_l[:,40:320,]])

    def depth_plus_visual(self,img_r,img_l,dep_r,dep_l):
        image = self.image_stich_manul(img_r,img_l)
        depth = self.image_stich_manul(dep_r,dep_l)

        min_val = np.min(depth)
        max_val = np.max(depth)
        depth_range = max_val - min_val
        depth8 = (255.0 / depth_range * (depth - min_val)).astype('uint8')
        depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
        depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)

        dep_visual = cv2.addWeighted(image, 0.5, depth_color, 0.5, 0)

        return dep_visual

    def encode(self,img):
        img = self.transform(img).unsqueeze(0).cuda()
        with torch.inference_mode():
            feature = self.dinov2_vits14(img)

        return feature
    def encode_image(self,img_r, img_l):
        image = self.image_stich_manul(img_r, img_l)
        feature = self.encode(image)
        return feature

    # feture is 1031
    def encode_image_state(self,img_r, img_l,state):
        feature = self.encode_image(img_r, img_l).cpu().squeeze()
        feature = np.concatenate([feature,state],axis=0)
        return feature

    def encode_image_depth(self,img_r, img_l,dep_r,dep_l):
        image = self.depth_plus_visual(img_r, img_l,dep_r,dep_l)
        feature = self.encode(image).cpu().squeeze()

        return feature

    def encode_image_depth_state(self,img_r, img_l,dep_r,dep_l,state):
        image = self.depth_plus_visual(img_r, img_l, dep_r, dep_l)
        feature = self.encode(image).cpu().squeeze()
        feature = np.concatenate([feature, state], axis=0)
        return feature





'''

base_path ='/home/carol/Project/off-lineRL/Data/logs-1206/05-12-2023-21-44-13/image/'
imgl_path = 'frontleft_fisheye_image/1701805533.83929562568664550781.png'
imgr_path = 'frontright_fisheye_image/1701805580.24047493934631347656.png'
imgdr_path = 'frontright_depth_in_visual_frame/1701805579.98835420608520507812.png'
imgdl_path = 'frontleft_depth_in_visual_frame/1701805580.28180360794067382812.png'

# open method used to open different extension image file
im_l = np.asarray(Image.open(base_path+imgl_path).convert('RGB'))
img_r = Image.open(base_path+imgr_path).convert('RGB')
img_r = np.asarray(img_r)
im_dl = np.asarray(Image.open(base_path+imgdl_path))
im_dr = np.asarray(Image.open(base_path+imgdr_path))
arm = np.zeros(7)

encode = Encoder()
feature = encode.encode_image_depth(im_l,img_r,im_dr,im_dl)
feature2 = encode.encode_image_depth_state(im_l,img_r,im_dr,im_dl,arm)
feature3 = encode.encode_image(im_l,img_r)
feature4 = encode.encode_image_state(im_l,img_r,arm)
print(feature2.shape)

'''