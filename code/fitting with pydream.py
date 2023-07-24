import numpy as np
import scipy.stats as st
import pandas as pd
from collections import OrderedDict
import warnings

from pydream.core import run_dream
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin
from scipy.stats import norm, lognorm, halfnorm

warnings.filterwarnings("error")


def filter_data(data, cell_type, use_mono_fb=True, use_mono_mp=False, day_intervals=[[3,7]]):
    
    # keep measurments of the required days
    _data = data[data.time_interval.isin([f'{i[0]}->{i[1]}' for i in day_intervals])].copy()
    
    # Start by filtering out mono-cultured cells and adding later the ones that we want
    data_to_fit = _data.loc[(_data.FB_type == cell_type) & (_data.initial_FB != 0) & (_data.initial_BM != 0)].copy()
    
    if use_mono_fb:
        data_to_fit = pd.concat([data_to_fit, _data.loc[(_data.FB_type == cell_type) & (_data.initial_BM == 0)]])
    
    if use_mono_mp:
        data_to_fit = pd.concat([data_to_fit, _data.loc[(_data.initial_FB == 0)]]) # all mono-cultured MPs
        
    return data_to_fit


def likelihood_log_counts(data, parameter_sample, parameter_names):

    data_for_FB = data[data['initial_FB'] != 0]
    data_for_BM = data[data['initial_BM'] != 0]

    # data likelihood
    like_FB = st.norm(loc=np.log(data_for_FB['#FB_final']), scale=1)
    like_BM = st.norm(loc=np.log(data_for_BM['#BM_final']), scale=1)

    # Model
    params = {name: value for name, value in zip(parameter_names, parameter_sample)}
    predicted_logF = 4*((params['pFF']*np.log1p(data_for_FB['#FB']) + params['pMF']*np.log1p(data_for_FB['#BM']))*(1-np.log1p(data_for_FB['#FB'])/params['KF']) - params['rF']) + np.log(data_for_FB['#FB'])
    predicted_logM = 4*((params['pMM']*np.log1p(data_for_BM['#BM']) + params['pFM']*np.log1p(data_for_BM['#FB']))*(1-np.log1p(data_for_BM['#BM'])/params['KM']) - params['rM']) + np.log(data_for_BM['#BM'])

    # Calculting the likelihhood of getting the simualtion values under the data distribution
    logp_FB = np.sum(like_FB.logpdf(predicted_logF))
    logp_BM = np.sum(like_BM.logpdf(predicted_logM))

    return logp_FB + logp_BM


if __name__ == '__main__':
    
    # name = 'nmf_dmem'
    name = 'nmf_cm'
    niterations = 10000
    nchains = 5
    
    parameter_priors = OrderedDict({
        'pFF': SampledParam(norm, loc=0, scale=1),
        'pMF': SampledParam(norm, loc=0, scale=1),
        'KF': SampledParam(norm, loc=15, scale=2),
        'rF': SampledParam(lognorm, s=1),

        'pMM': SampledParam(norm, loc=0, scale=1),
        'pFM': SampledParam(norm, loc=0, scale=1),
        'KM': SampledParam(norm, loc=15, scale=2),
        'rM': SampledParam(lognorm, s=1),
    })
    
    data = pd.read_csv('~/projects/cafs/code/pydream/full_model/data.csv')
    
    data_to_fit = filter_data(
        # data.loc[(data.medium == 'dmem10')],
        data.loc[(data.medium == 'cm10')],
        cell_type='nmf',
    )
    
    def likelihood_wrapper(parameter_sample):
        
        return likelihood_log_counts(data_to_fit, parameter_sample, parameter_priors.keys())
    
    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    sampled_params, log_ps = run_dream(
        parameters=list(parameter_priors.values()),
        likelihood=likelihood_wrapper,
        niterations=niterations,
        nchains=nchains,
        multitry=False,
        gamma_levels=4,
        adapt_gamma=True,
        history_thin=1,
        model_name=name,
        verbose=True
    )
    
    total_iterations = niterations

    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save(f'{name}_{nchains}chain_sampled_params_chain_{chain}_{total_iterations}', sampled_params[chain])
        np.save(f'{name}_{nchains}chain_logps_chain_{chain}_{total_iterations}', log_ps[chain])

    #Check convergence and continue sampling if not converged

    GR = Gelman_Rubin(sampled_params)
    print(f'At iteration: {total_iterations} GR = {GR}')
    np.savetxt(f'{name}_5chain_GelmanRubin_iteration_{total_iterations}.txt', GR)

    old_samples = sampled_params
    if np.any(GR>1.2):
        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
        converged = False
        while not converged:
            total_iterations += niterations
            sampled_params, log_ps = run_dream(
                parameters=list(parameter_priors.values()),
                likelihood=likelihood_wrapper,
                niterations=niterations,
                nchains=nchains,
                start=starts,
                multitry=False,
                gamma_levels=4,
                adapt_gamma=True,
                history_thin=1,
                model_name=name,
                verbose=True,
                restart=True
            )

            # Save sampling output (sampled parameter values and their corresponding logps).
            for chain in range(len(sampled_params)):
                np.save(f'{name}_{nchains}chain_sampled_params_chain_{chain}_{total_iterations}', sampled_params[chain])
                np.save(f'{name}_{nchains}chain_logps_chain_{chain}_{total_iterations}', log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
            GR = Gelman_Rubin(old_samples)
            print(f'At iteration: {total_iterations} GR = {GR}')
            np.savetxt(f'{name}_5chain_GelmanRubin_iteration_{total_iterations}.txt', GR)

            if np.all(GR<1.2):
                converged = True