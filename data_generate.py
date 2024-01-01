import os
from extract_dataset import extract_demonstrations
import numpy as np
import tqdm
from dino_encoder import Encoder
from PIL import Image

base_path =os.getcwd() + '/data/'

act_to_num = {
    'W': 0, 'A': 1, 'S': 2, 'D': 3,
    'w':4,'a':5,'s':6,'d':7,
    'R':8,'F':9,'I':10,'K':11,
    'U':12,'O':13,'J':14,'L':15,
     'q':16,'e':17,
}
def collect_data(fram_num = 3,data_path=None):
    # Create directory for dataset
    if data_path is not None:
        contexts = extract_demonstrations(data_path)
    else:
        contexts =np.load(base_path+"demo_all_img2.npy",allow_pickle=True)

    imgs_all = []
    gts_all = []
    actions_all = []
    rewards_all = []
    n_traj_per_context = len(contexts)
    image_feature = np.load(base_path+"enc_feature_all_img2.npy",allow_pickle=True)

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
                image_buffer.append(image)
            images = np.concatenate(image_buffer, axis=2)
            imgs_traj.append(images) #（32, 32, 3)

            state = traj["arm_state"]
            if state[2] ==0:
                state = np.delete(state, 2)
                print('deteleted 0')
            gts_traj.append(state) # (7)
            actions_traj.append([char_to_num[action]])

        if len(imgs_traj) != len(gts_traj):
            print(f'len of img is {len(imgs_traj)}, while len of state is {len(gts_traj)}')
            break
        imgs_all.append(np.array(imgs_traj))
        gts_all.append(np.array(gts_traj))
        actions_all.append(np.array(actions_traj))
        rewards_all.append(np.array(rewards_traj).astype(np.float32))

    np.save( base_path+"obs_all.npy", [imgs_all]) # image---video   19
    np.save(base_path+ "actions_all.npy", [actions_all])  #action
    np.save(base_path+"state7_all.npy",  [gts_all])#[gts_all]) # joint state
    np.save( base_path+"rewards_all.npy",  [rewards_all])#[rewards_all]) # reward,[0,1]
    np.save(base_path+"act_dic.npy", act_to_num)
    print(len(np.array(rewards_all, dtype=object)))

# for encoded feature
def collect_image(data_path=None):
    contexts = extract_demonstrations(base_path)
    #np.load("/home/carol/Project/Offline_RL/Data_collection/SpotDataExtract/data52/demo_all_img4.npy",allow_pickle=True)

    np.save('/home/carol/Project/Offline_RL/Data_collection/SpotDataExtract/data/demo_all_img2.npy', contexts)

    img_encoder = Encoder()


    imgs_all = []
    n_traj_per_context = len(contexts)

    for ind in tqdm.tqdm(range(0, n_traj_per_context)):

        imgs_traj = []
        traj_contexs, final_reward = contexts[ind]
        num_img = traj_contexs[0]['camera_images']

        for traj in traj_contexs:
            if num_img ==4:
                imgdl_path, imgl_path, imgdr_path, imgr_path = traj['camera_images']

            imgl_path, imgr_path = traj['camera_images']

            # open method used to open different extension image file
            img_l = np.asarray(Image.open( imgl_path).convert('RGB'))
            img_r = np.asarray(Image.open( imgr_path).convert('RGB'))

            if num_img == 4:
                im_dl = np.asarray(Image.open(imgdl_path))
                im_dr = np.asarray(Image.open(imgdr_path))
                feature = img_encoder.encode_image_depth(img_r=img_r, img_l=img_l, dep_r=im_dr, dep_l=im_dl)
            else:
                feature = img_encoder.encode_image(img_r, img_l)


            imgs_traj.append(feature.detach().cpu().numpy()) #(1, 1024)

        imgs_all.append(np.array(imgs_traj))
        print(f'\n{ind}th trajactory,  length is {len(imgs_traj)}\n')

    np.save(base_path+"enc_feature_all_img2.npy", np.array(imgs_all, dtype=object))  # image---video   19
    print(f'Done。 length is {len(imgs_all)}\n')

if __name__ == '__main__':

    collect_image()
    collect_data(fram_num=3)


