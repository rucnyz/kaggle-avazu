import os

import utils
from utils import *

sys.stdout = Logger('D:\\2014_mobilectr\\2_generate_fm_features.txt')
t0tv_mx_save = load(utils.tmp_data_path + 't0tv_mx.joblib_dat')
t0tv_mx = t0tv_mx_save['t0tv_mx']
click_values = t0tv_mx_save['click']
day_values = t0tv_mx_save['day']
print("t0tv_mx loaded with shape", t0tv_mx.shape)

t0 = load(utils.tmp_data_path + 't3a.joblib_dat')['t3a']
print("t0 loaded with shape", t0.shape)

vns = {}
_vns1 = ['app_or_web', 'banner_pos', 'C1', 'C15', 'C16', 'C20', 'C14',
         'cnt_device_ip_day_hour', 'cnt_device_ip_day_hour_prev',
         'cnt_device_ip_pday', 'dev_ip_cnt2', 'app_site_id', 'device_model']

vns['all_noid'] = _vns1
vns['all_withid'] = _vns1 + ['dev_id2plus']
vns['fm_5vars'] = ['app_or_web', 'banner_pos', 'C1', 'C15', 'device_model']
vns['all_but_ip'] = ['app_or_web', 'device_conn_type', 'C18', 'device_type',
                     'banner_pos', 'C1', 'C15', 'C16', 'hour1', 'as_category', 'C21',
                     'C19', 'C20', 'cnt_device_ip_day_hour', 'cnt_device_ip_pday',
                     'cnt_device_ip_day_hour_prev', 'cnt_device_ip_day_hour_next',
                     'dev_id_cnt2', 'dev_ip_cnt2', 'cnt_diff_device_ip_day_pday', 'C17',
                     'C14', 'device_model', 'as_domain', 'app_site_id', '_A_app_or_web',
                     '_A_C1', '_A_banner_pos', '_A_C16', '_A_C15', '_A_C18', '_A_C19',
                     '_A_C21', '_A_C20', '_A_C17', '_A_C14', 'as_model', 'dev_id2plus']

cmd_str = utils.fm_path + \
          ' -t 4 -s 8 -l 1e-5 ' + utils.tmp_data_path + '_tmp_2way_v.txt ' + utils.tmp_data_path + '_tmp_2way_t.txt'

day_bgn = 22
day_end = 32

fm_vecs = {}
path1 = utils.tmp_data_path
fn_t = path1 + '_tmp_2way_t.txt'
fn_v = path1 + '_tmp_2way_v.txt'
for day_v in range(day_bgn, day_end):
    fm_vecs[day_v] = {}
    for vns_name in vns.keys():
        vns2 = vns[vns_name]

        print(day_v, vns_name)
        t1 = t0.loc[:, ['click']].copy()

        idx_base = 0

        for vn in vns2:
            t1[vn] = t0[vn].values
            t1[vn] = np.asarray(t1[vn].astype('category').values.codes, np.int32) + idx_base
            idx_base = t1[vn].values.max() + 1
        # print '-'* 5, vn, idx_base

        print("to write data files ...")

        t1.loc[np.logical_and(day_values >= 21, day_values < day_v), :].to_csv(open(fn_t, 'w'), sep = '\t',
                                                                               header = False, index = False)
        t1.loc[day_values == day_v, :].to_csv(open(fn_v, 'w'), sep = '\t', header = False, index = False)

        print(cmd_str)
        os.system(cmd_str)

        print("load results ...")
        fm_predv = pd.read_csv(open(path1 + '_tmp_2way_v.txt.out', 'r'), header = None).loc[:, 0].values

        print("--- gini_norm:", gini_norm(fm_predv, click_values[day_values == day_v], None))

        fm_vecs[day_v][vns_name] = fm_predv
        print('=' * 60)

t2 = t0.loc[:, ['click']].copy()

nn = t2.shape[0]
for vns_name in vns.keys():
    t2[vns_name] = np.zeros(nn)
    for day_v in range(day_bgn, day_end):
        print(day_v, vns_name)
        t2.loc[day_values == day_v, vns_name] = fm_vecs[day_v][vns_name]

print("to save FM features ...")
dump(t2, tmp_data_path + 't2.joblib_dat')

t0tv_mx3 = np.concatenate([t0tv_mx[:, :43], t2.iloc[:, 1:].to_numpy()], axis = 1)
print("t0tv_mx3 generated with shape", t0tv_mx3.shape)

t0tv_mx_save = {'t0tv_mx': t0tv_mx3, 'click': click_values, 'day': day_values}
dump(t0tv_mx_save, utils.tmp_data_path + '/t0tv_mx3.joblib_dat')
sys.stdout.close()
