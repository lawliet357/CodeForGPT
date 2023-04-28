""" Description
This script is used as the main API for E3 python block.
function is started with e3_XXX
Return 1 if nothing to return, since e3 python block force to have at least 1 return argument
Need to install pandas, python, tensorflow first in E3 APP server
"""
""" API to write
def online_train_model(model_filepath), include validation
def visualize_model(model_filepath)
"""
import numpy as np
import pandas as pd
import pickle
import copy
import os
from tensorflow.keras.models import load_model
from tensorflow import split, reduce_sum
from tensorflow.nn import softplus
from tensorflow.math import log, square
from tensorflow import constant as constant_tf
from tensorflow import float32 as float32_tf

"""
Custom losses
"""
# import tensorflow_probability as tfp
def multi_prob_dist_loss_fn(beta, mu_global, sigma_global):
    """
    Arguments:
    1. beta: the coefficient to weight between custom loss and KL divergence
             loss = -likelihood + beta * KL
             For KL formula, see https://stats.stackexchange.com/questions/234757/how-to-use-kullback-leibler-divergence-if-mean-and-standard-deviation-of-of-two
    2. mu_global: The global mean of y, with shape (1, len(cy_meas))
    3. sigma_global: The global variance of y, with shape (1, len(cy_meas))
    """
    def custom_loss(y_true, y_pred):
        """
        ** This part of the code is the original one for multivariate proability distribution calculation using tfp
        We are waiting for the upgrade of tfp to replace the calculation formula below
        # split the y prediction into 2 subset: One for mean and one for log_sigma
        mu, log_sigma = tf.split(y_pred, 2, axis=-1)
        # Bound the variance (from best practice). The coefficient might need further optimization
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
        # Get the distribution
        dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        log_prob = dist.log_prob(y_true)
        # We want the multivariate probability is high given right mean and sigma.
        # Meaning the -log_prob should be minimize: max(prob) == min(-log_prob)
        loss = -tf.reduce_mean(log_prob, axis=-1) 
        return loss
        """
        """
        Code reference: https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e
        Algorithm reference: https://medium.com/analytics-vidhya/gaussian-mixture-models-with-tensorflow-probability-125315891c22
        mu, log_sigma = tf.split(y_pred, 2, axis=-1)
        log_sigma = tf.nn.softplus(log_sigma)
        n_dims = mu.shape[1]
        mse = -0.5 * tf.reduce_sum(tf.math.square((y_true-mu)/tf.math.exp(log_sigma)),axis=1)
        sigma_trace = -tf.reduce_sum(log_sigma, axis=1)
        log2pi = -0.5 * n_dims * tf.math.log(2 * np.pi)
        log_likelihood = mse + sigma_trace + log2pi
        """
        # Temporary formula
        mu, sigma = split(y_pred, 2, axis=-1)
        sigma = 1e-06 + softplus(sigma) # Force sigma to be positive. The sigma here is "variance", not std
        n_dims = mu.shape[1]
        mse = -0.5 * reduce_sum(square((y_true-mu)/sigma), axis=1)
        sigma_trace = -reduce_sum(log(sigma), axis=1)
        log2pi = -0.5 * n_dims * log(2 * np.pi)
        log_likelihood = mse + sigma_trace + log2pi
        # KL divergence
        mu_global_tf = constant_tf(mu_global, dtype=float32_tf)
        sigma_global_tf = constant_tf(sigma_global, dtype=float32_tf)
        KL_div = 0.5 * log(sigma_global_tf/sigma) + (sigma + (mu - mu_global_tf)**2)/(2*sigma_global_tf) - 0.5
        KL_div = beta * reduce_sum(KL_div, axis=1)

        return -log_likelihood + KL_div
    return custom_loss

def online_prediction(model_const, x_time, x_etime, nctxt, x_vctxt, x_nparam, x_vparam,
                            z_time, z_etime, z_vctxt, z_nparam, z_vparam,
                            x_vparam_base=None, z_vparam_base=None,
                            output_risk_control=False):
        """ 
        Desc:
            For single sample prediction
        Args:
        1. model_const: The model constant dictionary
        3. x_time (str): The process timestamp of the current run
        4. x_etime (str): The elpased seconds of the current run (from previous run of same chrono gorup)
        4. nctxt (str list): The name of the contexts
        5. x_vctxt (csv str): The contexts of the current run
        6. x_nparam (csv str): The parameter names of the current run
        7. x_vparam (csv str): The parameter values of the current run
        8. z_time (str list): The process timestamps of the ref runs
        9. z_etime (str list): The elapsed seconds of the ref runs
        10. z_vctxt (csv str list): The contexts of the ref runs
        11. z_nparam (csv str list): The parameter names of the ref runs
        12. z_vparam (csv str list): The parameter values of the ref runs. Need to match the order of z_nparam by sample
        13. horizon_key (str): The e3 horizon context key
        14. x_vparam_base (csv str): The parameter base values of the current run (for inc mode only)
        15. z_vparam_base (csv str list): The parameter base values of the reference runs (for inc mode only)
        16. output_risk_control (boolean): If predicted output is out of max/min, replace with Q3/Q1

        Returns:
        o (dbl list): The predicted y (model's output)
            for standard mode: predict absolute y
            for incremental mode: predict incremental y
        o_sigma (dbl list): The unpredictability (only for prob loss func)
        replace_o (boolean): The indices have been replaced with y-Q1/Q3
        feature_oos (boolean): True if x or z over feature max/min
        x_where_oos (int list): The feature indices of x oos 
        z_where_oos_1 (int list): The timestamp indices of z oos
        z_where_oos_2 (int list): The feature indices of z oos
        x_cont (dbl list): square of each x - x_median
        z_cont (dbl list of list): square of each z - z_median

        Remarks:
        1. For inc mode, normalize x_vparam and z_vparam with scale only, and base value with original min/max scalar
        2. Save z and x dataset for future online train (with dict using e3 context key as index)
        3. Save y dataset for future online train in another feature at post-metrology. (add oos label)
        4. Need to save dictionary for online training. i.e., dict[key] = new training data

        Remarks for e3 strategy:
        1. If inc mode, need to add 2 new columns for x and z base values
        2. Control system user data item need to add a new column to save elapsed time
        3. For inc mode, x_vparam and z_vparam should be calculate from the parameter value difference from the same chrono group
        4. z should be listed in ascending order chronologically, not descending
        5. len(z) should be ref_cnt+1, for inc_mode to calculate onehot context switching
        """
        # ============ Input format conversion and assertion ========================
        zn = len(z_nparam)
        if not model_const["inc_mode"]:
            assert len(z_nparam) == len(z_vparam) >= model_const["ref_cnt"], "z parameter definition list and value list do not match"
        else:
            assert len(z_nparam) == len(z_vparam) == len(z_vparam_base) >= model_const["ref_cnt+1"] # one more sample to calculate onehot diff for inc mode, see zx_onehot_inc
        x_vctxt = x_vctxt.split(',')
        df_z_vctxt = pd.Series(z_vctxt)
        df_z_vctxt = df_z_vctxt.str.split(pat=',',expand=True) # Expand context dataframe by seprate csv with comma
        assert df_z_vctxt.shape[1] == len(nctxt), "Context definition not equal to length of z contexts"
        df_z_vctxt.columns = nctxt # Replace column names with context def.
        assert len(z_time) >= model_const["ref_cnt"], "Not enough ref run information, the count of z is: " + str(len(z_time))
        assert len(x_vctxt) == len(nctxt), "Context definition not equal to length of x contexts"

        # ============ One-hot coding for x and z ========================
        # works on self object:
        # One-hot coding for x and z
        if model_const["onehot"]:
            onehot_x = dict()
            onehot_z = dict()
            for i,j in enumerate(model_const["onehot_ctxt_order"]):
                onehot_x[j] = model_const["onehot_dict"][j].loc[x_vctxt[nctxt.index(j)]].values.astype(float).reshape(1,-1) # onehot_dict is a dictionary saving onehot dataframe map
                onehot_z[j] = model_const["onehot_dict"][j].loc[df_z_vctxt[j]].values.astype(float) # get z onehot result at once by each context category
            onehot_x = np.concatenate([onehot_x[i] for i in model_const["onehot_ctxt_order"]], axis=1) # Concat x onehot result by context order
            onehot_z = np.concatenate([onehot_z[i] for i in model_const["onehot_ctxt_order"]], axis=1) # Concat z onehot result by context order
        
        # ==================== From data z ========================
        if not model_const["inc_mode"]: # for non-inc mode
            if model_const["onehot"]: zx_onehot = onehot_z
            if not model_const["meas_only"]:
                z_is_meas = list() # used for z is-meas index as an additional feature
            zy = list()
            zx = list()
            for i in range(zn):
                zy_sample = list()
                zx_sample = list()
                if not model_const["meas_only"]:
                    z_is_meas_sample = False
                nparam = z_nparam[i].split(',')
                vparam = z_vparam[i].split(',')
                for j in (model_const["cz_meas"]):
                    try: # For meas_only = False
                        zy_sample.append(vparam[nparam.index(j)])
                        if not model_const["meas_only"]:
                            z_is_meas_sample = np.True_
                    except:
                        if not model_const["meas_only"]: zy_sample.append(np.nan) # Accept NaN if meas_only = False
                        else: raise Exception('Cannot find z-measure value. Please check if cz_meas matches the measurement item of z_nparam')
                for j in (model_const["cz_process"]):
                    if j in model_const["cz_onehot"]: # onehot features are handled independently
                        continue
                    else:
                        zx_sample.append(vparam[nparam.index(j)])
                zy.append(np.array(zy_sample).reshape(1,-1))
                zx.append(np.array(zx_sample).reshape(1,-1))
                if not model_const["meas_only"]:
                    z_is_meas.append(z_is_meas_sample)
            zy = model_const["sc_zy"].transform(np.concatenate(zy, axis=0)) # normalize zy
            if not model_const["meas_only"]:
                zy = np.nan_to_num(zy) # replace non-measured samples to 0
                z_is_meas = np.array(z_is_meas).reshape(-1,1)
            if model_const["onehot"]: zx = model_const["sc_zx"].transform(np.concatenate([zx_onehot,np.concatenate(zx, axis=0)],axis=1)) # combine onehot features and normalize zx
            else: zx = model_const["sc_zx"].transform(np.concatenate(zx, axis=0))
            zte = np.array(z_etime).astype(float) / model_const["th"] # normalize zte
            zte[zte > 1] = 1 # clamp zte
            zte = zte.reshape(-1,1) # reshape zte to 2D
            if model_const["time_feature"]:
                zt = pd.Series(z_time + [x_time]) 
                zt = pd.to_datetime(zt).values.astype(float)/1e+9 # use pd parse date time string and convert to np
                zt[1:] = (zt[1:] - zt[:-1])
                zt[zt > model_const["td"]] = 1 # clamp
                zt = zt.cumsum()
                zt = (zt / zt.max()).reshape(-1,1) # normalize and reshape to 2D
                if model_const["meas_only"]:
                    z = np.expand_dims(np.concatenate([zy,zx,zte,zt[:-1]],axis=1),axis=0) # form z with 3D shape
                else:
                    z = np.expand_dims(np.concatenate([zy,z_is_meas,zx,zte,zt[:-1]],axis=1),axis=0) # form z with 3D shape
            else:
                if model_const["meas_only"]:
                    z = np.expand_dims(np.concatenate([zy,zx,zte],axis=1),axis=0) # form z with 3D shape
                else:
                    z = np.expand_dims(np.concatenate([zy,z_is_meas,zx,zte],axis=1),axis=0) # form z with 3D shape
        else: # for inc mode
            if model_const["onehot"]:
                zx_onehot_inc = onehot_z
                zx_onehot_inc[1:,:] = zx_onehot_inc[1:,:] - zx_onehot_inc[:-1,:] # calculate onehot diff, and we don't care the first sample
                zx_onehot_base = onehot_z.shift(period=1, axis=0) # get onehot base
            zy = list()
            zx = list()
            zy_base = list()
            zx_base = list()
            sc_zy_inc = copy.copy(model_const["sc_zy"])
            sc_zy_inc.min_ = 0
            sc_zx_inc = copy.copy(model_const["sc_zx"])
            sc_zx_inc.min_ = 0
            for i in range(zn):
                zy_sample = list()
                zy_base_sample = list()
                zx_sample = list()
                zx_base_sample = list()
                nparam = z_nparam[i].split(',')
                vparam = z_vparam[i].split(',')
                v_param_base = z_vparam_base[i].split(',')
                for j in (model_const["cz_meas"]):
                    zy_sample.append(vparam[nparam.index(j)])
                    zy_base_sample.append(v_param_base[nparam.index(j)])
                for j in (model_const["cz_process"]):
                    if j in model_const["cz_onehot"]: # onehot features are handled independently
                        continue
                    else:
                        zx_sample.append(vparam[nparam.index(j)])
                        zx_base_sample.append(v_param_base[nparam.index(j)])
                zy.append(np.array(zy_sample).reshape(1,-1))
                zy_base.append(np.array(zy_base_sample).reshape(1,-1))
                zx.append(np.array(zx_sample).reshape(1,-1))
                zx_base.append(np.array(zx_base_sample).reshape(1,-1))
            zy = sc_zy_inc.transform(np.concatenate(zy, axis=0)) # normalize zy inc
            zy_base = model_const["sc_zy"].transform(np.concatenate(zy_base, axis=0)) # normalize zy base
            if model_const["onehot"]:
                zx = sc_zx_inc.transform(np.concatenate([zx_onehot_inc,np.concatenate(zx, axis=0)],axis=1)) # combine onehot features and normalize zx inc
                zx_base = model_const["sc_zx"].transform(np.concatenate([zx_onehot_base,np.concatenate(zx_base, axis=0)],axis=1)) # combine onehot features and normalize zx base
            else:
                zx = sc_zx_inc.transform(np.concatenate(zx, axis=0))
                zx_base = model_const["sc_zx"].transform(np.concatenate(zx_base, axis=0)) 
            zte = np.array(z_etime).astype(float) / model_const["th"] # normalize zte
            zte[zte > 1] = 1 # clamp zte
            zte = zte.reshape(-1,1) # reshape zte to 2D
            if model_const["time_feature"]:
                zt = pd.Series(z_time + [x_time]) 
                zt = pd.to_datetime(zt).values.astype(float)/1e+9 # use pd parse date time string and convert to np
                zt[1:] = (zt[1:] - zt[:-1])
                zt[zt > model_const["td"]] = 1 # clamp
                zt = zt.cumsum()
                zt = (zt / zt.max()).reshape(-1,1) # normalize and reshape to 2D
                z = np.expand_dims(np.concatenate([zy,zy_base,zx,zx_base,zte,zt[:-1]],axis=1),axis=0) # form z with 3D shape
            else:
                z = np.expand_dims(np.concatenate([zy,zy_base,zx,zx_base,zte],axis=1),axis=0) # form z with 3D shape
        z = z[:,-model_const["ref_cnt"]:,:] # get the last ref_cnt samples of z

        # ==================== From data x ========================
        if not model_const["inc_mode"]: # for non-inc mode
            if model_const["onehot"]: x_onehot = onehot_x
            nparam = x_nparam.split(',')
            vparam = x_vparam.split(',')
            x_sample = list()
            for i in model_const["cx_process"]:
                if i in model_const["cx_onehot"]: # onehot features are handled independently
                    continue
                else:
                    x_sample.append(vparam[nparam.index(i)])
            x = np.array(x_sample).reshape(1,-1)
            if model_const["onehot"]: x = model_const["sc_x"].transform(np.concatenate([x_onehot,x],axis=1)) # combine onehot features and normalize x
            else: x = model_const["sc_x"].transform(x)
            xte = np.array([x_etime]).astype(float) / model_const["th"] # normalize xte
            xte[xte > 1] = 1 # clamp xte
            xte = xte.reshape(1,1) # reshape ste to 2D
            if model_const["time_feature"]:
                x = np.concatenate([x,xte,np.array([[1.0]])], axis=1) # the last normalized timestamp of x should always be 1.0
            else:
                x = np.concatenate([x,xte], axis=1) # the last normalized timestamp of x should always be 1.0
        else: # for inc mode
            if model_const["onehot"]:
                x_onehot_inc = onehot_x - onehot_z[-1:,:] # x - last z
                x_onehot_base = onehot_x
            sc_x_inc = copy.copy(model_const["sc_x"])
            sc_x_inc.min_ = 0
            nparam = x_nparam.split(',')
            vparam = x_vparam.split(',')
            vparam_base = x_vparam_base.split(',') # for x base values
            x_sample = list()
            x_sample_base = list()
            for i in model_const["cx_process"]:
                if i in model_const["cx_onehot"]: # onehot features are handled independently
                    continue
                else:
                    x_sample.append(vparam[nparam.index(i)])
                    x_sample_base.append(vparam_base[nparam.index(i)])
            x_inc = np.array(x_sample).reshape(1,-1)
            x_base = np.array(x_sample).reshape(1,-1)
            if model_const["onehot"]:
                x_inc = sc_x_inc.transform(np.concatenate([x_onehot_inc,x_inc],axis=1)) # combine onehot features and normalize x inc
                x_base = model_const["sc_x"].transform(np.concatenate([x_onehot_base,x_base],axis=1)) # combine onehot features and normalize x base
            else:
                x_inc = sc_x_inc.transform(x_inc)
                x_base = model_const["sc_x"].transform(x_base)
            
            xte = np.array([x_etime]).astype(float) / model_const["th"] # normalize xte
            xte[xte > 1] = 1 # clamp xte
            xte = xte.reshape(1,1) # reshape ste to 2D
            if model_const["time_feature"]:
                x = np.concatenate([x_inc,x_base,xte,np.array([[1.0]])], axis=1) # the last normalized timestamp of x should always be 1.0
            else:
                x = np.concatenate([x_inc,x_base,xte], axis=1) # the last normalized timestamp of x should always be 1.0
        
        # ========= Save online data for further training =======
        online_data = dict()
        online_data['z'] = z
        online_data['x'] = x
        c = list()
        for i in model_const["ctxt_name"]:
            c.append(x_vctxt[nctxt.index(i)]) # re-order context values
        online_data['c'] = c
        if model_const["inc_mode"]:
            online_data['yb'] = (zy_base[-1:,:])

        # ==================== Predict y ========================
        o = model_const["model"].predict([z,x])
        # Check feature oos
        x_oos = (x > model_const["quar"]['x']['100']) | (x < model_const["quar"]['x']['0'])
        x_where_oos = (np.where(x_oos))[0].tolist()
        if model_const["time_feature"]:
            model_const["quar"]['z']['100'][-1] = 1 
            model_const["quar"]['z']['0'][-1] = 0
            # z percentile was calculated based on the last timestamp
            #of z feature, hence the last z feature, which is timestamp (when time_feature=True), 
            # #will close to 1, but we only need to check z inside 0 to 1 is fine
            
        z_oos = (z[0,:,:] > model_const["quar"]['z']['100']) | (z[0,:,:] < model_const["quar"]['z']['0'])
        z_where_oos = np.where(z_oos)
        z_where_oos_1 = z_where_oos[0].tolist()
        z_where_oos_2 = z_where_oos[1].tolist()
        feature_oos = x_oos.sum() + z_oos.sum() > 0
        if feature_oos == True: feature_oos = True # convert from numpy bool_ to py bool
        else: feature_oos = False
        x_cont = np.square(x - model_const["quar"]['x']['50']).tolist()
        z_cont = np.square(z[0,:,:] - model_const["quar"]['z']['50']).tolist()
        # Replace o-max with y-Q3 and and o-min with Y-Q1 for risk control
        replace_o = list()
        if output_risk_control:
            overmax_idx = np.where(o > model_const["quar"]['y']['100'])[0]
            overmin_idx = np.where(o < model_const["quar"]['y']['0'])[0]
            o[overmax_idx] = model_const["quar"]['y']['75'][overmax_idx] #replace max with Q3
            o[overmin_idx] = model_const["quar"]['y']['25'][overmin_idx] # replace max with Q1
            replace_o = sorted(overmax_idx.tolist() + overmin_idx.tolist())
        if model_const["loss"] == 'prob':
            o_sigma = 1e-06 + np.log(1 + np.exp(o[:,len(model_const["cy_meas"]):])) # convert log_sigma -> softplus(log_sigma) -> sigma
            o = o[:,:len(model_const["cy_meas"])]
            online_data['o_sigma'] = o_sigma
            online_data['o'] = o
        else:
            o_sigma = (o * 0) # no unpredictability to return for loss func != non-prob
            online_data['o'] = o

        # ============= Inverse transform for y =================
        if model_const["inc_mode"]:
            sc_y = copy.copy(model_const["sc_y"])
            sc_y.min_ = 0 # inc mode, just keep scale and remove shifting (proven through derivation)
        else:
            sc_y = model_const["sc_y"]
        o = sc_y.inverse_transform(o) # Calculated prediction errors on measured wafers. Predicted output should not contain NaN.

        return o[0].tolist(), o_sigma[0].tolist(), online_data, replace_o, feature_oos, x_where_oos, z_where_oos_1, z_where_oos_2, x_cont, z_cont

def e3_online_prediction(model_const_filepath, keras_filepath, data_filepath, horizon_key,
                        x_time, x_etime, nctxt, x_vctxt, x_nparam, x_vparam,
                        z_time, z_etime, z_vctxt, z_nparam, z_vparam,
                        x_vparam_base=None, z_vparam_base=None,
                        output_risk_control=False, input_risk_control=False):
    """ 
    Desc:
        For single sample prediction
    Args:
        model_const_filepath (str): The filepath of model const dictionary to read
        keras_filepath (str): The keras model filepath to read
        data_fileptth (str): The online training data file path for saving
        horizon_key (str): The e3 horizon context key, used to save pickle file for online training data
        x_time (str): The process timestamp of the current run
        x_etime (str): The elpased seconds of the current run (from previous run of same chrono gorup)
        nctxt (str list): The name of the contexts
        x_vctxt (csv str): The contexts of the current run
        x_nparam (csv str): The parameter names of the current run
        x_vparam (csv str): The parameter values of the current run
        z_time (str list): The process timestamps of the ref runs
        z_etime (str list): The elapsed seconds of the ref runs
        z_vctxt (csv str list): The contexts of the ref runs
        z_nparam (csv str list): The parameter names of the ref runs
        z_vparam (csv str list): The parameter values of the ref runs. Need to match the order of z_nparam by sample
        x_vparam_base (csv str): The parameter base values of the current run (for inc mode only)
        z_vparam_base (csv str list): The parameter base values of the reference runs (for inc mode only)
        output_risk_control (boolean): If predicted output is out of max/min, replace with Q3/Q1
        input_risk_control (boolean): If x or z feature oos, then do not build online training dataset of this run

    Returns:
        o (dbl list): The predicted y (model's output)
            for standard mode: predict absolute y
            for incremental mode: predict incremental y
        o_sigma (dbl list): The unpredictability (only for prob loss func)
        replace_o (boolean): The indices have been replaced with y-Q1/Q3
        feature_oos (boolean): True if x or z over feature max/min
        x_where_oos (int list): The feature indices of x oos 
        z_where_oos_1 (int list): The timestamp indices of z oos
        z_where_oos_2 (int list): The feature indices of z oos
        x_cont (dbl list): square of each x - x_median
        z_cont (dbl list of list): square of each z - z_median

    Remarks:
    1. For inc mode, normalize x_vparam and z_vparam with scale only, and base value with original min/max scalar
    2. Save z and x dataset for future online train (with dict using e3 context key as index)
    3. Save y dataset for future online train in another feature at post-metrology. (add oos label)
    4. Need to save dictionary for online training. i.e., dict[key] = new training data

    Remarks for e3 strategy:
    1. If inc mode, need to add 2 new columns for x and z base values
    2. Control system user data item need to add a new column to save elapsed time
    3. For inc mode, x_vparam and z_vparam should be calculate from the parameter value difference from the same chrono group
    4. z should be listed in ascending order chronologically, not descending
    5. len(z) should be ref_cnt+1, for inc_mode to calculate onehot context switching
    6. Need to use latest model_const_filepath and keras_filepath to read.
    7. Need to check and delete old model files periodically
    """
    # Load model constants
    model_const_filepath = model_const_filepath.replace('.pickle','') + '.pickle'
    with open(model_const_filepath, 'rb') as handle:
        model_const = pickle.load(handle)
    # Load Keras model
    if model_const["loss"] == 'prob': model_const["model"] = \
                            load_model(keras_filepath, 
                            custom_objects={'custom_loss': multi_prob_dist_loss_fn(model_const["beta"], model_const["mu_global"], model_const["sigma_global"])})
    else: model_const["model"] = load_model(keras_filepath)
    # Do online prediction
    o, o_sigma, online_data, replace_o, feature_oos, x_where_oos, z_where_oos_1, z_where_oos_2, x_cont, z_cont =\
                        online_prediction(model_const, x_time, x_etime, nctxt, x_vctxt, x_nparam, x_vparam,
                        z_time, z_etime, z_vctxt, z_nparam, z_vparam,
                        x_vparam_base=x_vparam_base, z_vparam_base=z_vparam_base,
                        output_risk_control=output_risk_control)
    # save new online training data when feature_oos = False
    if (not input_risk_control) or (not feature_oos):
        if not os.path.exists(data_filepath):
            os.makedirs(data_filepath)
        with open(data_filepath+horizon_key+'.pickle', 'wb') as handle:
            pickle.dump(online_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    result = dict()
    result['o'] = o
    result['o_sigma'] = o_sigma
    result['model_const_filepath'] = model_const_filepath
    result['replace_o'] = replace_o
    result['feature_oos'] = feature_oos
    result['x_where_oos'] = x_where_oos
    result['z_where_oos_1'] = z_where_oos_1
    result['z_where_oos_2'] = z_where_oos_2
    result['x_cont'] = x_cont
    result['z_cont'] = z_cont
    return result