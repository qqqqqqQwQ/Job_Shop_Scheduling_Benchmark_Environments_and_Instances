def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path)]
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'solution_methods', 'DMOFJSSP_DRL'))
import numpy as np
import torch
import time
import os
import argparse
from solution_methods.DMOFJSSP_DRL.Params import configs
from solution_methods.DMOFJSSP_DRL.models.actor_critic import Job_Mch_Actor
from solution_methods.DMOFJSSP_DRL.agent_utils import greedy_select_action_mch, sample_select_action_mch
from solution_methods.DMOFJSSP_DRL.poisson_instance import Instance_Generator
import random
import time
from solution_methods.DMOFJSSP_DRL.Configuration_Env2.MODFJSSP_Env_2 import MODFJSSP
from scheduling_environment import jobShop
device = torch.device(configs.device)

class PPO:
    def __init__(self,
                lr,
                gamma,
                k_epochs,
                eps_clip,
                n_j,
                n_m
                ):

        self.policy_job_mch = Job_Mch_Actor(n_j=n_j,
                                            n_m=n_m,
                                            n_layers_fea = configs.num_mlp_layers_fea,
                                            input_dim_fea = configs.input_dim_fea,
                                            hidden_dim_fea = configs.hidden_dim_fea,
                                            out_dim_fea = configs.out_dim_fea,
                                            n_layers_actor = configs.num_mlp_layers_actor,
                                            input_dim_actor = configs.input_dim_actor,
                                            hidden_dim_actor = configs.hidden_dim_actor,
                                            out_dim_actor = configs.out_dim_actor,
                                            device=device)

    
def main(run_times, e_ave_Test, New_insert, machine, DDT_Test,seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    #50_20_8, 50_20_12, 50_20_16, 100_30_8, 100_30_12, 100_30_16, 200_40_8, 200_40_12, 200_40_16
    # e_ave_Test=50  # 50, 100, 200
    # New_insert=20 # 20, 30, 40
    # machine=8  # 8, 12, 16
    # DDT_Test=1.5    

    Processing_time, A, D, M_num, Op_num, J, O_num, J_num = Instance_Generator(machine, e_ave_Test, New_insert, DDT_Test)
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
                n_j=J_num,
                n_m=M_num)
  

    filepath = 'solution_methods/DMOFJSSP_DRL/Saved_network'
    # 选择泊松分布训练的策略模型
    filepath = os.path.join(filepath, f'MODFJSSP_P_e_{e_ave_Test}_DDT_{DDT_Test}')
    if not os.path.isdir(filepath):
        raise FileNotFoundError(f"Model directory not found: {filepath}")
    filepaths = get_imlist(filepath)
    if not filepaths:
        raise FileNotFoundError(f"No model files found in {filepath}")
    model_path = filepaths[0]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    ppo.policy_job_mch.load_state_dict(torch.load(model_path))

    result=[]
    times=[]
    envs=[]
    # ppo中的actor采用随机采样动作 sample_select_action_mch，所以每次结果不一样，要重复20次。若贪婪策略则一次即可
    for k in range(run_times):
        t1 = time.time()
        env = MODFJSSP(J_num, M_num, max(Op_num))
        js=jobShop.JobShop()
        js.set(Processing_time, A, D, M_num, Op_num, O_num, J_num)
        fea1, fea2, fea3, candidate, mask, mch_time, job_time, ci, Lateness_ave, U_ave = env.reset(Processing_time, A, D, M_num, Op_num, J, O_num, J_num)
        while True:
            fea1_tensor = torch.from_numpy(np.copy(fea1)).to(device).type(torch.float32)
            fea2_tensor = torch.from_numpy(np.copy(fea2)).to(device).type(torch.float32)
            fea3_tensor = torch.from_numpy(np.copy(fea3)).to(device).type(torch.float32)                        
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device).type(torch.int64)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device).reshape(1,-1)

            mch_time_tensor = torch.from_numpy(np.copy(mch_time)).to(device).type(torch.float32).expand(J_num, M_num) 
            job_time_tensor = torch.from_numpy(np.copy(job_time)).to(device).type(torch.float32).expand(J_num, M_num)

            ci_tensor = torch.from_numpy(np.copy(ci)).to(device).type(torch.float32).expand(J_num, M_num)

            with torch.no_grad():
                pi_a_mch, a_mch_pool = ppo.policy_job_mch(fea1=fea1_tensor,
                                                        fea2=fea2_tensor,
                                                        fea3=fea3_tensor,
                                                        mch_time=mch_time_tensor,
                                                        job_time=job_time_tensor,
                                                        ci=ci_tensor,
                                                        mask=mask_tensor)

                action, mch = sample_select_action_mch(pi_a_mch, candidate_tensor, M_num)

                row = action // max(Op_num)
                col = action % max(Op_num)
                if col < Op_num[row]:
                    op_id = sum(Op_num[:row]) + col
                    op = js.get_operation(op_id)
                    if op is not None and mch in op.processing_times:
                        js.get_machine(mch).add_operation_to_schedule(op, op.processing_times[mch], js.sequence_dependent_setup_times)

                fea1, fea2, fea3, reward, done, candidate, mask, mch_time, job_time, ci, Lateness_ave, U_ave = env.step(action, mch)

            if env.done():
                Makespan = max(torch.squeeze(job_time).numpy())
                U_ave_1 = 1/U_ave            
                result.append([Makespan,U_ave_1,Lateness_ave])
                envs.append(js)

                break  
        t2 = time.time()
        # print('Makespan:', Makespan)
        # print('U_ave_1:', U_ave_1)
        # print('Lateness_ave:', Lateness_ave)   
        times.append(t2-t1)

    result=np.array(result)
    times=np.array(times)
    # np.savez('result_{}_{}_{}_save.npz'.format(e_ave_Test, New_insert, machine),result)
    # np.savez('times_{}_{}_{}_save.npz'.format(e_ave_Test, New_insert, machine),times)
    Times_mean=np.mean(times)
    # print('Times_mean:',Times_mean)
    # print()
    return envs,result,times

if __name__ == '__main__':
    from plotting.drawer import draw_performance_3d
    e_ave_Test = 20   # 泊松训练均值
    New_insert = 20   # 插入作业数
    machine = 8       
    DDT_Test = 1.2    # 泊松训练交期系数
    run_times=20
    seed=30
    envs,result,times=main(run_times, e_ave_Test, New_insert, machine, DDT_Test,seed)
    # print(envs[0].makespan)
    # print(envs[0].U_ave_1)
    # print(envs[0].Lateness_ave)
    draw_performance_3d(envs)