import os, time, argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict
import json
from AGNN_model import AGNN
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.0005, type=float,
					help="learning rate.")
parser.add_argument("--dropout", default=0.5, type=float,
					help="dropout rate.")
parser.add_argument("--batch_size", default=128, type=int,
					help="batch size when training.")
parser.add_argument("--gpu", default="0", type=str,
					help="gpu card ID.")
parser.add_argument("--epochs", default=20, type=str,
					help="training epoches.")
parser.add_argument("--clip_norm", default=5.0, type=float,
					help="clip norm for preventing gradient exploding.")
parser.add_argument("--embed_size", default=30, type=int, help="embedding size for users and items.")
parser.add_argument("--attention_size", default=50, type=int, help="embedding size for users and items.")
parser.add_argument("--item_layer1_nei_num", default=10, type=int)
parser.add_argument("--user_layer1_nei_num", default=10, type=int)
parser.add_argument("--vae_lambda", default=1, type=int)

#################################evaluation############################################
def metrics(model, test_dataloader):
    label_lst, pred_lst = [], []
    rmse, mse, mae = 0,0,0
    count = 0
    for batch_data in test_dataloader:
        user = torch.LongTensor(batch_data[0]).cuda()
        item = torch.LongTensor(batch_data[1]).cuda()
        label = torch.FloatTensor(batch_data[2]).cuda()
        user_self_cate = torch.LongTensor(batch_data[3]).cuda()
        user_onehop_id = torch.LongTensor(batch_data[4]).cuda()
        user_onehop_cate = torch.LongTensor(batch_data[5]).cuda()
        item_self_cate, item_self_director, item_self_writer, item_self_star, item_self_country = torch.LongTensor(
            batch_data[6])[:, 0:6].cuda(), torch.LongTensor(batch_data[6])[:, 6:9].cuda(), torch.LongTensor(
            batch_data[6])[:, 9:12].cuda(), torch.LongTensor(batch_data[6])[:, 12:15].cuda(), torch.LongTensor(
            batch_data[6])[:, 15:].cuda()
        item_onehop_id = torch.LongTensor(batch_data[7]).cuda()
        item_onehop_cate, item_onehop_director, item_onehop_writer, item_onehop_star, item_onehop_country = torch.LongTensor(
            batch_data[8])[:, :, 0:6].cuda(), torch.LongTensor(batch_data[8])[:, :, 6:9].cuda(), torch.LongTensor(
            batch_data[8])[:, :, 9:12].cuda(), torch.LongTensor(batch_data[8])[:, :, 12:15].cuda(), torch.LongTensor(
            batch_data[8])[:, :, 15:].cuda()

        prediction, recon_loss, kl_loss = model(user, item, user_self_cate, user_onehop_id, user_onehop_cate, item_self_cate,
                           item_self_director, item_self_writer, item_self_star, item_self_country, item_onehop_id,
                           item_onehop_cate, item_onehop_director, item_onehop_writer, item_onehop_star,
                           item_onehop_country, mode = mode)
        prediction = prediction.cpu().data.numpy()
        prediction = prediction.reshape(prediction.shape[0])
        label = label.cpu().numpy()
        my_rmse = np.sum((prediction - label) ** 2)
        my_mse = np.sum((prediction - label) ** 2)
        my_mae = np.sum(np.abs(prediction - label))
        # my_rmse = torch.sqrt(torch.sum((prediction - label) ** 2) / FLAGS.batch_size)
        rmse+=my_rmse
        mse+=my_mse
        mae+=my_mae
        count += len(user)
        label_lst.extend(list([float(l) for l in label]))
        pred_lst.extend(list([float(l) for l in prediction]))

    my_mse = mse/count
    my_rmse = np.sqrt(rmse/count)
    my_mae = mae/count
    return my_rmse, my_mse, my_mae, label_lst, pred_lst
###########################################################################


def get_data_list(ftrain, batch_size):  #完整训练测试数据
    f = open(ftrain, 'r')
    train_list = []
    for eachline in f:
        eachline = eachline.strip().split('\t')
        u, i, l = int(eachline[0]), int(eachline[1]), float(eachline[2])
        train_list.append([u, i, l])
    num_batches_per_epoch = int((len(train_list) - 1) / batch_size) + 1
    return num_batches_per_epoch, train_list

def get_batch_instances(train_list, user_feature_dict, item_feature_dict, item_director_dict, item_writer_dict, item_star_dict, item_country_dict, batch_size, user_nei_dict, item_nei_dict, shuffle=True):
    #是否打乱数据，再用yield分块送入
    num_batches_per_epoch = int((len(train_list) - 1) / batch_size) + 1
    def data_generator(train_list):
        data_size = len(train_list)
        user_feature_arr = np.array(list(user_feature_dict.values()))
        max_user_cate_size = user_feature_arr.shape[1]

        item_genre_arr = np.array(list(item_feature_dict.values())) #len=6 ,0
        item_director_arr = np.array(list(item_director_dict.values())) #len=3 ,6
        item_writer_arr = np.array(list(item_writer_dict.values())) #len=3, 9
        item_star_arr = np.array(list(item_star_dict.values())) #len=3, 12
        item_country_arr = np.array(list(item_country_dict.values()))   #len=8, 15

        item_feature_arr = np.concatenate([item_genre_arr, item_director_arr, item_writer_arr, item_star_arr, item_country_arr], axis=1)
        max_item_cate_size = item_feature_arr.shape[1]

        item_layer1_nei_num = FLAGS.item_layer1_nei_num
        user_layer1_nei_num = FLAGS.user_layer1_nei_num

        if shuffle == True:
            np.random.shuffle(train_list)
        train_list = np.array(train_list)

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            current_batch_size = end_index - start_index

            u = train_list[start_index: end_index][:, 0].astype(np.int)
            i = train_list[start_index: end_index][:, 1].astype(np.int)
            l = train_list[start_index: end_index][:, 2]

            i_self_cate = np.zeros([current_batch_size, max_item_cate_size], dtype=np.int)
            i_onehop_id = np.zeros([current_batch_size, item_layer1_nei_num], dtype=np.int)
            i_onehop_cate = np.zeros([current_batch_size, item_layer1_nei_num, max_item_cate_size], dtype=np.int)

            u_self_cate = np.zeros([current_batch_size, max_user_cate_size], dtype=np.int)
            u_onehop_id = np.zeros([current_batch_size, user_layer1_nei_num], dtype=np.int)
            u_onehop_cate = np.zeros([current_batch_size, user_layer1_nei_num, max_user_cate_size], dtype=np.int)

            for index, each_i in enumerate(i):
                i_self_cate[index] = item_feature_arr[each_i]    #item_self_cate

                tmp_one_nei = item_nei_dict[each_i][0]
                tmp_prob = item_nei_dict[each_i][1]
                if len(tmp_one_nei) > item_layer1_nei_num:  #re-sampling
                    tmp_one_nei = np.random.choice(tmp_one_nei, item_layer1_nei_num, replace=False, p=tmp_prob)
                elif len(tmp_one_nei) < item_layer1_nei_num:
                    tmp_one_nei = np.random.choice(tmp_one_nei, item_layer1_nei_num, replace=True, p=tmp_prob)
                tmp_one_nei[-1] = each_i

                i_onehop_id[index] = tmp_one_nei    #item_1_neigh
                i_onehop_cate[index] = item_feature_arr[tmp_one_nei]  #item_1_neigh_cate

            for index, each_u in enumerate(u):
                u_self_cate[index] = user_feature_dict[each_u]  # item_self_cate

                tmp_one_nei = user_nei_dict[each_u][0]
                tmp_prob = user_nei_dict[each_u][1]
                if len(tmp_one_nei) > user_layer1_nei_num:  # re-sampling
                    tmp_one_nei = np.random.choice(tmp_one_nei, user_layer1_nei_num, replace=False, p=tmp_prob)
                elif len(tmp_one_nei) < user_layer1_nei_num:
                    tmp_one_nei = np.random.choice(tmp_one_nei, user_layer1_nei_num, replace=True, p=tmp_prob)
                tmp_one_nei[-1] = each_u

                u_onehop_id[index] = tmp_one_nei  # user_1_neigh
                u_onehop_cate[index] = user_feature_arr[tmp_one_nei]  # user_1_neigh_cate

            yield ([u, i, l, u_self_cate, u_onehop_id, u_onehop_cate, i_self_cate, i_onehop_id, i_onehop_cate])
    return data_generator(train_list)

if __name__ == '__main__':
    #item cold start
    f_info = '../ml100k/uiinfo.pkl'
    f_neighbor = '../ml100k/neighbor_aspect_extension_2_zscore_ics_uuii_0.20.pkl'
    f_train = '../ml100k/ics_train.dat'
    f_test = '../ml100k/ics_val.dat'
    f_model = '../ml100k/agnn_ics_'
    mode = 'ics'

    """# user cold start
    f_info = '../ml100k/uiinfo.pkl'
    f_neighbor = '../ml100k/neighbor_aspect_extension_2_zscore_ucs_uuii.pkl'
    f_train = '../ml100k/ucs_train.dat'
    f_test = '../ml100k/ucs_val.dat'
    f_model = '../ml100k/agnn_ucs_'
    mode = 'ucs'"""

    """# warm start
    f_info = '../ml100k/uiinfo.pkl'
    f_neighbor = '../ml100k/neighbor_aspect_extension_2_zscore_warm_uuii.pkl'
    f_train = '../ml100k/warm_train.dat'
    f_test = '../ml100k/warm_val.dat'
    f_model = '../ml100k/agnn_warm_'
    mode = 'warm'"""


    FLAGS = parser.parse_args()
    print("\nParameters:")
    print(FLAGS.__dict__)

    with open(f_neighbor, 'rb') as f:
        neighbor_dict = pickle.load(f)
    user_nei_dict = neighbor_dict['user_nei_dict']
    item_nei_dict = neighbor_dict['item_nei_dict']
    director_num = neighbor_dict['director_num']
    writer_num = neighbor_dict['writer_num']
    star_num = neighbor_dict['star_num']
    country_num = neighbor_dict['country_num']

    item_director_dict = neighbor_dict['item_director_dict']    #dict[i]=[x,x,x]
    item_writer_dict = neighbor_dict['item_writer_dict']        #dict[i]=[x,x,x]
    item_star_dict = neighbor_dict['item_star_dict']            #dict[i]=[x,x,x]
    item_country_dict = neighbor_dict['item_country_dict']      #dict[i]=[x,x,x,x,x,x,x,x]

    with open(f_info, 'rb') as f:
        item_info = pickle.load(f)
    user_num = item_info['user_num']
    item_num = item_info['item_num']
    gender_num = item_info['gender_num']
    age_num = item_info['age_num']
    occupation_num = item_info['occupation_num']
    genre_num = item_info['genre_num']
    user_feature_dict = item_info['user_feature_dict']  #gender, age, occupation    dict[u]=[x,x,x]
    item_feature_dict = item_info['item_feature_dict']  #genre                      dict[i]=[x,x,x,x,x,x]

    print("user_num {}, item_num {}, gender_num {}, age_num {}, occupation_num {}, genre_num {}, director_num {}, writer_num {}, star_num {}, country_num {}, mode {} ".format(user_num, item_num, gender_num, age_num, occupation_num, genre_num, director_num, writer_num, star_num, country_num, mode))

    train_steps, train_list = get_data_list(f_train, batch_size=FLAGS.batch_size)
    test_steps, test_list = get_data_list(f_test, batch_size=FLAGS.batch_size)

    model = AGNN(user_num, item_num, gender_num, age_num, occupation_num, genre_num, director_num, writer_num, star_num, country_num, FLAGS.embed_size, FLAGS.attention_size, FLAGS.dropout)
    model.cuda()

    loss_function = torch.nn.MSELoss(size_average=False)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FLAGS.lr, weight_decay=0.001)

    writer = SummaryWriter()  # For visualization
    #f_loss_curve = open('tmp_loss_curve.txt', 'w')
    best_rmse = 5

    count = 0
    for epoch in range(FLAGS.epochs):
        #tmp_main_loss, tmp_vae_loss = [], []
        model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_dataloader = get_batch_instances(train_list, user_feature_dict, item_feature_dict, item_director_dict, item_writer_dict, item_star_dict, item_country_dict,  batch_size=FLAGS.batch_size, user_nei_dict=user_nei_dict, item_nei_dict=item_nei_dict, shuffle=True)

        for idx, batch_data in enumerate(train_dataloader): #u, i, l, u_self_cate, u_onehop_id, u_onehop_rating, u_onehop_cate, i_self_cate, i_onehop_id, i_onehop_cate
            user = torch.LongTensor(batch_data[0]).cuda()
            item = torch.LongTensor(batch_data[1]).cuda()
            label = torch.FloatTensor(batch_data[2]).cuda()
            user_self_cate = torch.LongTensor(batch_data[3]).cuda()
            user_onehop_id = torch.LongTensor(batch_data[4]).cuda()
            user_onehop_cate = torch.LongTensor(batch_data[5]).cuda()
            item_self_cate, item_self_director, item_self_writer, item_self_star, item_self_country = torch.LongTensor(batch_data[6])[:, 0:6].cuda(), torch.LongTensor(batch_data[6])[:, 6:9].cuda(), torch.LongTensor(batch_data[6])[:, 9:12].cuda(), torch.LongTensor(batch_data[6])[:, 12:15].cuda(), torch.LongTensor(batch_data[6])[:, 15:].cuda()
            item_onehop_id = torch.LongTensor(batch_data[7]).cuda()
            item_onehop_cate, item_onehop_director, item_onehop_writer, item_onehop_star, item_onehop_country = torch.LongTensor(batch_data[8])[:, :, 0:6].cuda(), torch.LongTensor(batch_data[8])[:, :, 6:9].cuda(), torch.LongTensor(batch_data[8])[:, :, 9:12].cuda(), torch.LongTensor(batch_data[8])[:, :, 12:15].cuda(), torch.LongTensor(batch_data[8])[:, :, 15:].cuda()

            model.zero_grad()
            prediction, recon_loss, kl_loss = model(user, item, user_self_cate, user_onehop_id, user_onehop_cate, item_self_cate, item_self_director, item_self_writer, item_self_star, item_self_country, item_onehop_id, item_onehop_cate, item_onehop_director, item_onehop_writer, item_onehop_star, item_onehop_country, mode='train')

            label = Variable(label)

            main_loss = loss_function(prediction, label)
            loss = main_loss + FLAGS.vae_lambda * (recon_loss + kl_loss)

            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), FLAGS.clip_norm)
            optimizer.step()
            writer.add_scalar('data/loss', loss.data, count)
            count += 1

        tmploss = torch.sqrt(loss / FLAGS.batch_size)
        print(50 * '#')
        print('epoch: ', epoch, '     ', tmploss.detach())

        model.eval()
        print('time = ', time.time() - start_time)
        test_dataloader = get_batch_instances(test_list, user_feature_dict, item_feature_dict, item_director_dict, item_writer_dict, item_star_dict, item_country_dict, batch_size=FLAGS.batch_size, user_nei_dict=user_nei_dict, item_nei_dict=item_nei_dict, shuffle=False)
        rmse, mse, mae, label_lst, pred_lst = metrics(model, test_dataloader)
        print('test rmse,mse,mae: ', rmse,mse,mae)

        """if (rmse < best_rmse):
            best_rmse = rmse
            f_name = f_model + str(best_rmse)[:7] + '.dat' #f_model + str(best_rmse)[:7] + '.dat'
            #torch.save(model, f_name)
            f = open(f_name, 'w')
            res_dict = {}
            res_dict['label'] = label_lst
            res_dict['pred'] = pred_lst
            json.dump(res_dict, f)
            f.close()
            print('save result ok')"""