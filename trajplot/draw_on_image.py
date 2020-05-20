import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def convert(data_path, homomatrix_path):
    # read world data and homomatrix
    try:
        df = pd.read_table(data_path, names=['frame_id', 'ped_id', 'gt_x', 'gt_y', 'pred_x', 'pred_y'])
        matrix = pd.read_table(homomatrix_path, sep='\t', header=None)
    except:
        print('cant find files!\n')
        return None

    gt_df = np.array(df[['gt_x', 'gt_y']])
    pred_df = np.array(df[['pred_x', 'pred_y']])
    one = np.ones(gt_df.shape[0])
    pos_gt = np.column_stack((gt_df, one))
    pos_pred = np.column_stack((pred_df, one))

    # inverse homograph matrix
    h = np.array(matrix, dtype=float)
    H = np.linalg.pinv(h)

    # convert
    pos_img_gt = np.dot(H, pos_gt.T)
    pos_img_pred = np.dot(H, pos_pred.T)
    img_gt_df = pd.DataFrame(pos_img_gt.T, columns=['gt_x', 'gt_y', 'z'])
    img_pred_df = pd.DataFrame(pos_img_pred.T, columns=['pred_x', 'pred_y', 'z'])

    # Normalize pixels
    img_gt_df['gt_x'] = img_gt_df['gt_x'] / img_gt_df.z.values
    img_gt_df['gt_y'] = img_gt_df['gt_y'] / img_gt_df.z.values
    img_pred_df['pred_x'] = img_pred_df['pred_x'] / img_pred_df.z.values
    img_pred_df['pred_y'] = img_pred_df['pred_y'] / img_pred_df.z.values
    print('Nomarlized pixels:\n', img_gt_df.head())

    # Merger into new Dataframe
    df1 = df[['frame_id', 'ped_id']]
    df2 = img_gt_df[['gt_x', 'gt_y']]
    df3 = img_pred_df[['pred_x', 'pred_y']]
    frames = [df1, df2, df3]
    img_df = pd.concat(frames, axis=1)
    print('Converted data:\n', img_df.head())
    return img_df


def img_plot(img_path, save_dir, frame_id, gt_x, gt_y, pred_x, pred_y):
    switch = False
    img_name = str(frame_id) + '.jpg'
    if 'hotel' or 'eth' in img_path:
        print('eth, switch.\n')
        switch = True
    img = img_path + img_name
    try:
        im = plt.imread(img)
    except:
        print("cant find this frame!", img, '\n')
        return None

    if switch:  # eth data need switch the coordination
        gt_y, gt_x = gt_x, gt_y
        pred_y, pred_x = pred_x, pred_y

    implot = plt.imshow(im)
    plt.axis('off')
    plt.plot(pred_x, pred_y, c='r')
    plt.plot(pred_x[-1], pred_y[-1], c='y', marker='x', markersize=12)
    plt.plot(gt_x, gt_y, c='b')
    plt.plot(gt_x[-1], gt_y[-1], c='y', marker='x', markersize=12)
    fig = plt.gcf() # get current fig to avoid of empty img
    # adjust margin to make plt fit img
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.show()
    if save_dir is not None:
        save = input("save this plot? yes or no:");
        if save in ['yes', 'y']:

            fig.savefig(save_dir + 'plot' + img_name)
        else:
            print("Dicard plot!")

def img_draw(img_df, img_path, save_dir, obs_len, pred_len):
    ped_id = img_df['ped_id'].unique().tolist()  # get ped_id
    print('ped_id:\n', ped_id)
    seq_len = obs_len+pred_len
    for idx in ped_id:
        #get target frame
        frame_id = img_df[img_df.ped_id==idx].frame_id.values[obs_len-1]
        target_df = img_df[img_df.ped_id==idx][['gt_x', 'gt_y', 'pred_x', 'pred_y']]
        arr = np.array(target_df)
        gt_x, gt_y, pred_x, pred_y = arr[:,0], arr[:,1], arr[:,2], arr[:,3]
        if len(gt_x) > seq_len:
            gt_x = gt_x[:seq_len]
            gt_y = gt_y[:seq_len]
            pred_x = [pred_x[obs_len+(seq_len*(i-1)):seq_len*i] for i in range(1, int(pred_x.size/seq_len + 1))]
            pred_y = [pred_y[obs_len + (seq_len * (i - 1)):seq_len * i] for i in range(1, int(pred_y.size / seq_len) + 1)]
            pred_x = np.array(pred_x).reshape(-1)
            pred_y = np.array(pred_y).reshape(-1)

        img_plot(img_path, save_dir, frame_id, gt_x, gt_y, pred_x, pred_y)


if __name__ == '__main__':
    # parameters
    world_path = '/home/want/Project/SoTrajectory/model/Flow_based/eth.txt'  # world coordinations file  header: frame_id  pred_id  gt_x  gt_y  pred_x  pred_y
    matrix_path = '/home/want/Desktop/homography_matrix /eth_univ.txt'  # path of homograph matrix
    img_path = '/home/want/Desktop/biwi_eth/'  # picture that to be plot traj
    save_dir = './'  # output dir of plotted image
    obs_len = 8  # time step that begin to plot
    pred_len = 8
    img_df = convert(world_path, matrix_path)
    if img_df is not None:
        img_draw(img_df, img_path, save_dir, obs_len, pred_len)