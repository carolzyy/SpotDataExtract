import os
import sys
from extract_dataset import extract_demonstrations
from eval_data import extract_images
import cv2
import numpy as np
import tqdm
import torch
from torchvision import transforms
import ssl
import h5py

char_to_num = {
    'W': 0, 'A': 1, 'S': 2, 'D': 3,
    'w':4,'a':5,'s':6,'d':7,
    'R':8,'F':9,'I':10,'K':11,
    'U':12,'O':13,'J':14,'L':15,
     'q':16,'e':17,
     'M':18,'N':19}

def image_merge(path_list):
    img_dl_path, img_l_path, img_dr_path, img_r_path = path_list
    img_l = cv2.imread(img_l_path, cv2.IMREAD_UNCHANGED)
    img_r = cv2.imread(img_r_path, cv2.IMREAD_UNCHANGED)
    img_dl = cv2.imread(img_dl_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    img_dr = cv2.imread(img_dr_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    #img_dr = cv2.resize(img_dr, (640, 480), interpolation=cv2.INTER_NEAREST)
    #downsampling
    new_shape = (60,80)
    img_l = cv2.resize(img_l, new_shape, interpolation=cv2.INTER_AREA)
    img_r = cv2.resize(img_r, new_shape, interpolation=cv2.INTER_AREA)
    img_dr = cv2.resize(img_dr, new_shape, interpolation=cv2.INTER_AREA)
    img_dl = cv2.resize(img_dl, new_shape, interpolation=cv2.INTER_AREA)


    img_R = cv2.addWeighted(img_r, 0.5, img_dr, 0.5, 0)
    img_L = cv2.addWeighted(img_l, 0.5, img_dl, 0.5, 0)

    image = np.concatenate((img_L,img_R), axis=2)


    return image

def image_stich_manul(path_list):

    img_l_path, img_r_path = path_list
    img_l = cv2.imread(img_l_path)
    img_r = cv2.imread(img_r_path)

    #img_dr = cv2.resize(img_dr, (640, 480), interpolation=cv2.INTER_NEAREST)
    #downsampling
    #new_shape = (60,80)
    #img_l = cv2.resize(img_l, new_shape, interpolation=cv2.INTER_AREA)
    #img_r = cv2.resize(img_r, new_shape, interpolation=cv2.INTER_AREA)

    # Concatenate images horizontally [:,:400,]
    merged_image = cv2.hconcat([img_r[:,40:320,],img_l[:,40:320,]])


    return merged_image


def encode_image(img):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    ssl._create_default_https_context = ssl._create_unverified_context
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dinov2_vits14.eval()
    device = 'cuda'
    dinov2_vits14.cuda()

    image = transform(img).unsqueeze(dim=0).to(device)
    with torch.inference_mode():
        feature = dinov2_vits14(image)
    #feature.reshape(32,32,1)
    return feature.reshape(32,32,1).detach().cpu().numpy(),feature.detach().cpu().numpy()

def collect_data(fram_num = 3):
    # Create directory for dataset
    #contexts = extract_demonstrations('/home/carol/Project/off-lineRL/Data/logs-curtain/')
    contexts =np.load("/home/carol/Project/off-lineRL/dataset/data/curtain/demo_all_img2.npy",allow_pickle=True)
    #contexts = contexts[::2]

    imgs_all = []
    gts_all = []
    actions_all = []
    rewards_all = []
    n_traj_per_context = len(contexts)
    image_feature = np.load("/home/carol/Project/off-lineRL/dataset/data/curtain/enc_feature_all_img2.npy",allow_pickle=True)

    for ind in tqdm.tqdm(range(0, n_traj_per_context)):

        imgs_traj = []
        gts_traj = []

        actions_traj = []
        traj_contexs, final_reward = contexts[ind]

        if len(traj_contexs)==0:
            continue
        print(f'{ind}th trajactory,  length is {len(traj_contexs)}')
        image_contex = image_feature[ind]

        rewards_traj = np.zeros_like(traj_contexs)  # 0
        rewards_traj[-1] = float(final_reward)

        for ind_,traj in enumerate(traj_contexs):
            action = traj['action']

            image_buffer = []
            for i in range(fram_num):
                index = ind_+i
                if index >= len(traj_contexs):
                    index = len(traj_contexs) -1
                image = image_contex[index].reshape(32,32,1)
                #image = image_stich_manul(traj_contexs[index]['camera_images'])
                #image,_ = encode_image(image)
                image_buffer.append(image)
            images = np.concatenate(image_buffer, axis=2)
            imgs_traj.append(images) #（32, 32, 3)

            state = traj["arm_state"]
            if state[2] ==0:
                state = np.delete(state, 2)
            gts_traj.append(state) # (7)
            actions_traj.append([char_to_num[action]])

        if len(imgs_traj) != len(gts_traj):
            print(f'len of img is {len(imgs_traj)}, while len of state is {len(gts_traj)}')
            break
        imgs_all.append(np.array(imgs_traj))
        gts_all.append(np.array(gts_traj))
        actions_all.append(np.array(actions_traj))
        rewards_all.append(np.array(rewards_traj).astype(np.float32))



    np.save( "/home/carol/Project/off-lineRL/dataset/data/curtain/obs_all.npy", [imgs_all]) # image---video   19
    np.save( "/home/carol/Project/off-lineRL/dataset/data/curtain/actions_all.npy", [actions_all])  #action
    np.save("/home/carol/Project/off-lineRL/dataset/data/curtain/state7_all.npy",  [gts_all])#[gts_all]) # joint state
    np.save( "/home/carol/Project/off-lineRL/dataset/data/curtain/rewards_all.npy",  [rewards_all])#[rewards_all]) # reward,[0,1]
    print(len(np.array(rewards_all, dtype=object)))

# for encoded feature
def collect_image(data_path):
    # Create directory for dataset
    contexts = extract_demonstrations(base_path = data_path)
    #contexts = np.load("/home/carol/Project/off-lineRL/dataset/data/curtain/demo_all.npy")
    np.save("/home/carol/Project/off-lineRL/dataset/data/curtain/demo_all_img2.npy", contexts)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    ssl._create_default_https_context = ssl._create_unverified_context
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dinov2_vits14.eval()
    device = 'cuda'
    dinov2_vits14.cuda()


    imgs_all = []
    n_traj_per_context = len(contexts)

    for ind in tqdm.tqdm(range(0, n_traj_per_context)):

        imgs_traj = []
        traj_contexs, final_reward = contexts[ind]

        for traj in traj_contexs:

            image = image_stich_manul(traj['camera_images'])
            image = transform(image).unsqueeze(dim=0).to(device)
            with torch.inference_mode():
                feature = dinov2_vits14(image)
            imgs_traj.append(feature.detach().cpu().numpy()) #(1, 1024)


        imgs_all.append(np.array(imgs_traj))
        print(f'\n{ind}th trajactory,  length is {len(imgs_traj)}\n')

    np.save("/home/carol/Project/off-lineRL/dataset/data/curtain/enc_feature_all_img2.npy", np.array(imgs_all, dtype=object))  # image---video   19
    print(f'Done。 length is {len(imgs_all)}\n')

# for image itself
def encode_images():
    contexts = extract_demonstrations()
    np.save("/home/carol/Project/off-lineRL/dataset/data/curtain/demo_all_img2.npy", contexts)
    n_traj_per_context = len(contexts)
    imgs_all = []
    for ind in tqdm.tqdm(range(0, n_traj_per_context)):

        imgs_traj = []
        traj_contexs, final_reward = contexts[ind]
        traj_contexs = traj_contexs[2:]  # start after i

        for traj in traj_contexs:

            image = image_stich_manul(traj['camera_images'])
            imgs_traj.append(image)

        imgs_all.append(np.array(imgs_traj))


    np.save("/data/curtain/enc_feature_all_img2.npy", [imgs_all])  # image---video   19

def collect_mul_image(fram_num=3):
    base_path = '/home/carol/Project/off-lineRL/dataset/data/curtain'

    # contexts = extract_demonstrations('/home/carol/Project/off-lineRL/Data/logs-curtain/')
    contexts = np.load(base_path+'/demo_all_img2.npy', allow_pickle=True)

    imgs_all = []
    n_traj_per_context = len(contexts)
    image_feature = np.load(base_path+"/enc_feature_all_img2.npy",
                            allow_pickle=True)
    for ind in tqdm.tqdm(range(0, n_traj_per_context)):

        imgs_traj = []

        traj_contexs, final_reward = contexts[ind]
        print(f'{ind}th trajactory,  length is {len(traj_contexs)}')
        if len(traj_contexs) == 0:
            print('there is a wrong traj!!!!!')
            continue
        image_contex = image_feature[ind]

        rewards_traj = np.zeros_like(traj_contexs)  # 0

        rewards_traj[-1] = float(final_reward)

        for ind_, traj in enumerate(traj_contexs):

            image_buffer = []
            for i in range(fram_num):
                index = ind_ + i
                if index >= len(traj_contexs):
                    index = len(traj_contexs) - 1
                image = image_contex[index].reshape(32, 32, 1)
                image_buffer.append(image)
            images = np.concatenate(image_buffer, axis=2)
            imgs_traj.append(images)  # （32, 32, 3)


        imgs_all.append(np.array(imgs_traj))

    np.save(base_path+"/obs_all.npy", [imgs_all])  # image---video   19
    print(len(np.array(imgs_all, dtype=object)))

if __name__ == '__main__':
    #collect_data(fram_num=3)
    #collect_image(data_path='/home/carol/Project/off-lineRL/Data/logs-curtain/')
    collect_data(fram_num = 3)


