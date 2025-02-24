import utils
from utils import *

sys.stdout = Logger('D:\\2014_mobilectr\\4_final_output.txt')
t0tv_mx_save = load(utils.tmp_data_path + 't0tv_mx.joblib_dat')
click_values = t0tv_mx_save['click']
day_values = t0tv_mx_save['day']
site_id_values = t0tv_mx_save['site_id']
print("t0tv_mx loaded")

day_test = 30
if utils.tvh == 'Y':
    day_test = 31

# RandomForest model output
rf_pred = load(utils.tmp_data_path + 'rf_pred_v.joblib_dat')
print("RF prediction loaded with shape", rf_pred.shape)

# GBDT (xgboost) model output
xgb_pred = load(utils.tmp_data_path + 'xgb_pred_v.joblib_dat')
print("xgb prediction loaded with shape", xgb_pred.shape)

# GBDT+LR model output
ctr = 0
lr_pred = 0
for i in [1, 2, 3, 4]:
    lr_pred += pd.read_csv(open(utils.tmp_data_path + 'lr__r%d_v.txt.out' % i, 'r'), header = None).loc[:, 0].values
    ctr += 1
lr_pred /= ctr
print("LR prediction :", lr_pred.shape)

# factorization machine model output
ctr = 0
fm_pred = 0
for i in [51, 52, 53, 54]:
    fm_pred += pd.read_csv(open(utils.tmp_data_path + 'fm__r%d_v.txt.out' % i, 'r'), header = None).loc[:, 0].values
    ctr += 1
fm_pred /= ctr
print("FM prediction:", fm_pred.shape)
# rf:0.4006; xgb:0.4004; lr:0.4098; fm:0.4021;
# 可以注意到其他不变，xgb越大，效果会上升到0.4001；但在继续增大则会下降到0.4004
# 有待调参
blending_w = {'rf': .25, 'xgb': .25, 'lr': .25, 'fm': .25}

total_w = 0

pred = rf_pred * blending_w['rf']
total_w += blending_w['rf']
pred += xgb_pred * blending_w['xgb']
total_w += blending_w['xgb']
pred += lr_pred * blending_w['lr']
total_w += blending_w['lr']
pred += fm_pred * blending_w['fm']
total_w += blending_w['fm']

pred /= total_w

if utils.tvh == 'Y':
    # create submission
    predh_raw_avg = pred
    site_ids_h = site_id_values[day_values == 31]
    tmp_f1 = site_ids_h == '17d1b03f'
    predh_raw_avg[tmp_f1] *= .13 / predh_raw_avg[tmp_f1].mean()
    predh_raw_avg *= .161 / predh_raw_avg.mean()

    sub0 = pd.read_csv(open(utils.raw_data_path + 'sampleSubmission', 'r'))
    pred_h_str = ["%.4f" % x for x in predh_raw_avg]
    sub0['click'] = pred_h_str
    fn_sub = utils.tmp_data_path + 'sub_sample' + str(utils.sample_pct) + '.csv.gz'
    import gzip

    sub0.to_csv(gzip.open(fn_sub, 'w'), index = False)
    print("=" * 80)
    print("Training complted and submission file " + fn_sub + " created.")
    print("=" * 80)
else:
    # validate using day30
    print("Training completed!")
    print("=" * 80)
    print("logloss of blended prediction:", logloss(pred, click_values[day_values == day_test]))
    print("=" * 80)
sys.stdout.close()
